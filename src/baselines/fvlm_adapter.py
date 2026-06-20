"""fVLM 백본 어댑터 — upstream을 충실하게 감싼 얇은 래퍼.

`third_party/fvlm/` (alibaba-damo-academy/fvlm)을 감쌈. 이 모델은 CT-RATE에서
**장기별 대조 정렬**을 수행하는 BLIP 기반 VLM. CT-CLIP처럼 단순히
`encode_image(volume) -> (B, dim)` 을 노출하는 백본이 아님 — forward 경로가
볼륨과 함께 장기-분할 마스크를 요구하고 장기별 피처를 반환. 따라서 이 어댑터는
CT-CLIP이 사용하는 encode_image/encode_text/tokenize 통일 계약을 의도적으로 노출하지 않음.

장기-마스크 사전조건: fVLM은 forward 시 `(volume, organ_mask)` 쌍을 기대.
CT-RATE는 마스크를 제공하지 않지만, 미리 계산된 TotalSegmentator 출력이
/workspace/datasets/datasets/CT-RATE/dataset/ts_seg/ts_total/ 에 이미 존재
(스캔당 하나의 다중 레이블 NIfTI). src/baselines/fvlm_preprocess.py:load_ct_and_mask_for_local
이 이를 읽어 {lung:1, heart:2, esophagus:3, aorta:4} 로 리매핑. 이 어댑터는
TotalSegmentator를 직접 호출하지 않음 — 마스크는 호출자가 제공하는 데이터 파이프라인 입력.

이 어댑터가 제공하는 것:

    backbone.model                  -> 실제 BlipPretrain 인스턴스
    backbone.prepare_text_feat(...) -> model.prepare_text_feat 패스스루
    backbone.forward_test_win(...)  -> model.forward_test_win 패스스루
    backbone.encode_organ_texts(..) -> 장기별 자유텍스트 → (len(organs), 256) 텍스트 임베딩
    backbone.encode_organs(img,msk) -> 장기별 ROI 이미지 임베딩 {organ: (256,)} (eval 충실)

사용자(또는 태스크 측 LightningModule)가 upstream 메서드를 직접 호출.
표준 평가 파이프라인은 third_party/fvlm/eval.py 참고.

왜 `BlipPretrain.from_config()`를 쓰지 않는가? upstream의 `from_config`가
개발자 전용 MAE-pretrain 경로(`/storage/guoruizhe/.../mae_pretrain_vit_base.pth`)를
하드코딩하고, 비-None `cfg`도 요구하기 때문 (`cfg.get(...)` 호출). 대신
third_party/fvlm/eval.py 의 평가측 조립 방식과 동일하게 컴포넌트를 수동으로 조립한 뒤,
MAE init을 덮어쓰는 릴리스된 CT-RATE 체크포인트를 로드.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from src.baselines.fvlm_preprocess import (
    FVLM_ROI_SIZE,
    center_crop_organ,
    divisible_pad_end,
)
from src.data.fvlm_organ_report import ORGANS, normal_organ_template

_FVLM_REPO: Final[Path] = Path("/workspace/third_party/fvlm")

# upstream `XBertEncoder.from_config`가 사용하는 HF id (fvlm/lavis/models/med.py:1402 참고).
_HF_TEXT_ENCODER: Final[str] = "microsoft/BiomedVLP-CXR-BERT-specialized"
if str(_FVLM_REPO) not in sys.path:
    sys.path.insert(0, str(_FVLM_REPO))


# __init__.py가 무거운 비-모델 의존성(decord, medspacy, totalsegmentator,
# transformers==4.25)을 끌어오는 lavis 서브패키지 이름 목록.
# import 시점에 스텁 처리하여 `lavis/__init__.py`가 깔끔하게 완료되게 함.
_HEAVY_LAVIS_SUBPACKAGES: Final[tuple[str, ...]] = (
    "lavis.datasets",
    "lavis.datasets.builders",
    "lavis.processors",
    "lavis.tasks",
)


def _stub_heavy_lavis_subpackages(lavis_root: Path) -> None:
    """sys.modules에 빈 `lavis.{datasets,processors,tasks}` 모듈을 미리 등록하여
    실제 `lavis/__init__.py`가 decord/medspacy/transformers==4.25 import를
    트리거하는 대신 스텁(no-op)을 찾도록 함.

    `__path__`를 보존하여 실제로 필요한 깊은 서브모듈
    (예: `lavis.processors.base_processor`)은 스텁 등록 후에도 Python의
    일반 서브모듈 해석으로 계속 로드될 수 있게 함.
    """
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415

    for pkg in _HEAVY_LAVIS_SUBPACKAGES:
        if pkg in sys.modules:
            continue
        rel = pkg.split(".", 1)[1].replace(".", "/")
        stub = types.ModuleType(pkg)
        stub.__path__ = [str(lavis_root / rel)]
        stub.__all__ = []
        sys.modules[pkg] = stub


CKPT_DIR: Final[Path] = Path("/workspace/data/checkpoints/fvlm")
DEFAULT_CKPT: Final[Path] = CKPT_DIR / "pretrained_ct_rate.pt"

# DUPLICATION INTENTIONAL — upstream `BlipPretrain.from_config`의 ViT kwargs를
# 그대로 미러. 출처: third_party/fvlm/blip_pretrain.py:333-340.
# upstream이 언젠가 config 파일을 제공하면, src/baselines/maisi.py처럼
# ConfigParser로 전환할 것.
_VIT_KWARGS: Final[dict] = {
    "in_channels": 1,
    "img_size": (112, 256, 352),
    "patch_size": (16, 16, 32),
    "num_classes": 0,
    "dropout_rate": 0.1,
    "qkv_bias": True,
}


class FVLMBackbone:
    """fVLM의 `BlipPretrain`을 감싼 얇은 래퍼.

    `nn.Module`이 아님 — 감싸진 `BlipPretrain`이 실제 학습 가능한 Module이며
    `.model`로 접근 가능. 여기서 `nn.Module`을 상속하면 `@property model`이
    `nn.Module.__getattr__`에 가려짐(`_parameters / _buffers / _modules`만 탐색).
    따라서 단순 클래스가 가장 깔끔한 컨테이너.

    Args:
        ckpt_path: fVLM 사전학습 CT-RATE 체크포인트 경로.
        device_str: 빌드 후 모델을 배치할 디바이스.
        load_weights: False이면 릴리스된 체크포인트 없이 빌드
            (내부 ViT는 랜덤 초기화 — MAE-pretrained init은 의도적으로
            건너뛰는 upstream 하드코딩 경로 필요).
    """

    def __init__(
        self,
        ckpt_path: str | Path = DEFAULT_CKPT,
        device_str: str = "cpu",
        load_weights: bool = True,
    ) -> None:
        self._ckpt_path = Path(ckpt_path)
        self._device_str = device_str
        self._load_weights = load_weights
        self._model: nn.Module | None = None

    def _ensure_built(self) -> None:
        if self._model is not None:
            return
        # fVLM은 자체 `lavis` 포크(`lavis/models/`)를 번들로 포함. 같은 프로세스에
        # 다른 `lavis`가 이미 import돼 있으면 sys.modules["lavis"]가 fVLM 서브모듈
        # 경로를 가릴 수 있음. `lavis*` 항목을
        # 모두 제거하고 sys.path[0]에 이 어댑터의 레포를 우선 등록하여
        # 이후 import가 fVLM으로 해석되게 함.
        for _name in [m for m in sys.modules if m == "lavis" or m.startswith("lavis.")]:
            sys.modules.pop(_name, None)
        while str(_FVLM_REPO) in sys.path:
            sys.path.remove(str(_FVLM_REPO))
        sys.path.insert(0, str(_FVLM_REPO))

        # `from lavis...` import 전에 무거운 lavis 서브패키지를 스텁 처리.
        # fVLM은 Salesforce LAVIS 프레임워크를 번들링했는데, `lavis/__init__.py`가
        # `lavis.datasets.builders`, `lavis.processors`, `lavis.tasks`를 즉시 import하여
        # decord(비디오 IO), medspacy(NLP), totalsegmentator,
        # 우리 환경(4.46)과 충돌하는 transformers==4.25 핀을 끌어옴.
        # 필요한 것은 `lavis.models`(와 거기서 참조하는 가벼운
        # `lavis.processors.base_processor`)뿐이므로, 나머지 세 곳에 빈 스텁을
        # 미리 등록하되 `__path__`를 설정하여 깊은 서브모듈 접근은 여전히 디스크로 해석되게 함.
        _stub_heavy_lavis_subpackages(_FVLM_REPO / "lavis")

        from lavis.models.blip_models.vit import ViT  # noqa: PLC0415
        from lavis.models.med import XBertEncoder  # noqa: PLC0415
        from transformers.models.bert.configuration_bert import BertConfig  # noqa: PLC0415

        # `BlipBase.__init__`(fVLM의 BlipPretrain 부모)이 `transformers<4.27`을
        # assert — Salesforce LAVIS 2022 시대의 가드. 우리 환경은 4.46.
        # 이 가드를 유발한 실제 API 브레이킹은 fVLM이 사용하지 않는
        # BLIP의 토크나이저/디코더 경로에 있음: fVLM forward는
        # `XBertEncoder.forward_text`만 사용하며, CT-CLIP 스모크 테스트에서
        # 4.46 아래 정상 작동을 이미 검증. 부모 __init__을 no-op-equivalent
        # (nn.Module.__init__만 호출)로 교체하여 생성자가 진행되게 함.
        from lavis.models.blip_models import blip as _blip  # noqa: PLC0415
        from lavis.models.base_model import BaseModel as _LavisBase  # noqa: PLC0415
        from transformers import BertTokenizer  # noqa: PLC0415

        _blip.BlipBase.__init__ = _LavisBase.__init__

        # `BlipBase.init_tokenizer`가 cwd에 그 이름의 *로컬* 디렉터리를 기대하며
        # `BertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")`를 하드코딩.
        # 정규 HF id로 리다이렉트하여 HF_HOME 캐시를 통해 해석되게 함.
        _blip.BlipBase.init_tokenizer = classmethod(
            lambda _cls: BertTokenizer.from_pretrained(_HF_TEXT_ENCODER)
        )

        # lavis `from lavis.models import *`가 __init__ 시점에 Salesforce의 원본
        # `BlipPretrain`을 'blip_pretrain_vit' 이름으로 이미 등록. fVLM의 루트
        # `blip_pretrain.py`는 자체 `BlipPretrain` 서브클래스(장기-인식, vision_projs +
        # query_tokens 포함)를 정의하고 같은 레지스트리 이름 사용 → KeyError.
        # 슬롯을 해제하여 fVLM 서브클래스가 깔끔하게 등록되게 함.
        from lavis.common.registry import registry  # noqa: PLC0415

        registry.mapping["model_name_mapping"].pop("blip_pretrain_vit", None)

        from blip_pretrain import BlipPretrain  # noqa: PLC0415

        image_encoder = ViT(**_VIT_KWARGS)

        # third_party/fvlm/lavis/models/med.py:1393-1410 (XBertEncoder.from_config)
        # 본문을 미러하되, 망가진 cfg/path-replace 처리는 제외.
        # upstream은 fvlm 디렉터리 내 microsoft/BiomedVLP-CXR-BERT-specialized
        # 로컬 클론을 가리키는 `med_config_path`가 있는 cfg를 기대.
        # 대신 HF id로 직접 해석 (HF_HOME이 한 번 캐시함).
        #
        # vanilla `BertConfig.from_pretrained(...)`가 설정하지 않지만
        # `lavis/models/med.py:BertEmbeddings/BertSelfAttention/BertEncoder`가
        # 무조건 읽는 네 개의 lavis 커스텀 config 속성을 주입.
        # 릴리스된 fVLM ckpt state-dict에서 추론한 값
        # (`token_type_embeddings` 없음, `crossattention.*` 키 없음):
        # type-embedding과 cross-attention 모두 이 모델에서는 OFF.
        med_config = BertConfig.from_pretrained(_HF_TEXT_ENCODER)
        med_config.add_type_embeddings = False
        med_config.add_cross_attention = False
        med_config.encoder_width = med_config.hidden_size
        med_config.fusion_layer = med_config.num_hidden_layers
        text_encoder = XBertEncoder.from_pretrained(
            _HF_TEXT_ENCODER, config=med_config, add_pooling_layer=False
        )
        for name, param in text_encoder.named_parameters():
            if "cls.predictions" in name:
                param.requires_grad = False

        model = BlipPretrain(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=None,
            tie_enc_dec_weights=False,
        )

        if self._load_weights:
            if not self._ckpt_path.is_file():
                raise FileNotFoundError(
                    f"fVLM checkpoint missing at {self._ckpt_path}. "
                    f"See docs/vlm_baselines_runbook.md for the gdrive download recipe."
                )
            ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                import logging  # noqa: PLC0415

                logging.getLogger(__name__).info(
                    "fVLM load: %d missing, %d unexpected keys",
                    len(missing),
                    len(unexpected),
                )

        model.eval()
        model.to(self._device_str)
        self._model = model

    # --- 얇은 패스스루 (통일 계약 없음 — 모듈 docstring 참고) -----

    @property
    def model(self) -> nn.Module:
        """지연 빌드되는 fVLM `BlipPretrain` 인스턴스."""
        self._ensure_built()
        return self._model

    def prepare_text_feat(self, test_items, length=None):
        """`BlipPretrain.prepare_text_feat` 패스스루.

        third_party/fvlm/eval.py:270 의 eval 시점 호출을 미러.
        """
        return self.model.prepare_text_feat(test_items, length=length)

    @torch.no_grad()
    def forward_test_win(self, *args, **kwargs):
        """`BlipPretrain.forward_test_win` 패스스루.

        third_party/fvlm/eval.py:318 의 eval 시점 호출을 미러.
        `(images, masks, text_feat_dict, ...)` 필요 — upstream 시그니처 참고.
        """
        return self.model.forward_test_win(*args, **kwargs)

    @torch.no_grad()
    def encode_organ_texts(
        self,
        organ_text: dict[str, str],
        organs: tuple[str, ...] = ORGANS,
    ) -> torch.Tensor:
        """장기별 자유텍스트 → fVLM 이미지-정렬 텍스트 임베딩 `(len(organs), 256)`.

        `prepare_text_feat`(neg/pos 분류 전용)와 달리, 장기당 자유텍스트 한 문장을
        fVLM의 ITC 텍스트 경로로 인코딩해 ctgen conditioning용 임베딩을 만든다.
        입력 `organ_text`는 `src.data.fvlm_organ_report.build_organ_text` 출력
        (모든 장기 키가 항상 존재; 누락 장기는 정상 템플릿으로 채워짐).

        각 organ에 대해 `blip_pretrain.py`의 학습-시 텍스트 경로를 그대로 미러한다.
        DUPLICATION INTENTIONAL — 출처: third_party/fvlm/blip_pretrain.py:212-226.
        """
        m = self.model  # 지연 빌드 트리거
        device = m.text_encoder.device  # blip_pretrain.py:534
        feats = []
        for organ in organs:
            text = organ_text[organ]
            tmpl = normal_organ_template(
                organ
            )  # "{organ} shows no significant abnormalities."
            # blip_pretrain.py:212-214 — 정상 템플릿 접두 제거(정확히 템플릿이면 유지).
            # findings는 있고 impression 없는 장기 → findings만 인코딩되도록.
            if text.startswith(tmpl) and text != tmpl:
                text = text.replace(tmpl, "")
            tokens = m.tokenizer(  # blip_pretrain.py:216-222
                text,
                padding="max_length",
                truncation=True,
                max_length=m.max_txt_len,  # 175
                return_tensors="pt",
            ).to(device)
            text_output = m.text_encoder.forward_text(tokens)  # blip_pretrain.py:224
            cls = text_output.last_hidden_state[:, 0, :]  # (1, 768) CLS 토큰
            feat = F.normalize(m.text_proj(cls), dim=-1)  # (1, 256) proj + L2 norm
            feats.append(feat)
        return torch.cat(feats, dim=0)  # (len(organs), 256)

    @torch.no_grad()
    def encode_organs(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        organs: tuple[str, ...] = ORGANS,
        *,
        crop_size: tuple[int, int, int] = FVLM_ROI_SIZE,
    ) -> dict[str, torch.Tensor]:
        """전처리된 전체 흉부 볼륨 → 장기별 fVLM ROI 이미지 임베딩 `{organ: (256,)}`.

        third_party/fvlm/eval.py:284-338 의 장기별 추론 루프를 충실 재현한다. 업스트림은
        **슬라이딩 윈도우가 아니라** (eval.py:301-307 dense_patch_slices 는 dead code)
        장기마다 그 장기를 중심에 둔 윈도우 하나만 추론에 쓴다: `center_crop_organ`
        (장기 중심 crop, roi=(112,288,352)) + `divisible_pad_end` (ViT 패치 배수) →
        업스트림 `forward_test_win(..., skip_organ=organ_id)` 1회.

        forward_test_win 은 `organ_feat_dict[organ_name]` 에 순수 이미지 피처
        `F.normalize(vision_projs[organ_id](attention(query_tokens[organ_id], organ_tokens)))`
        `(1, 256)` 를 저장한다(blip_pretrain.py:459-482). 빈 `organ_logits`/`text_feat_dict`
        로 호출하면 텍스트 대조(484-492)는 건너뛰므로 이 이미지 피처만 그대로 받아온다.
        (재구현 없이 업스트림 함수 재사용 — saliency 의 grad-enabled 사본과 달리 grad 불필요.)

        Args:
            image: (1, 1, D, H, W) float — `load_ct_and_mask_for_local` 출력(전체 볼륨).
            mask:  (1, 1, D, H, W) int  — multilabel {0,1,2,3,4}.
            organs: 처리할 장기(기본 4개). mask 에 없는 장기는 결과에서 빠진다.
            crop_size: 장기별 윈도우 최소 크기 = eval roi_size (112, 288, 352).

        Returns:
            `{organ_name: (256,) float tensor}` — mask 에 존재하는 장기만
            (eval 의 organ_feat_dict[fid] 미러; organ-absent 는 eval.py:295 처럼 skip).
        """
        m = self.model  # 지연 빌드 트리거
        device = next(m.parameters()).device
        image = image.to(device)  # (1, 1, D, H, W)
        mask = mask.to(device)  # (1, 1, D, H, W)
        feats: dict[str, torch.Tensor] = {}
        for organ in organs:
            organ_id = m.organs.index(organ)  # 0-based, eval 과 동일
            label = organ_id + 1
            whole = int(
                (mask == label).sum()
            )  # 전체 볼륨 내 장기 복셀 수 (eval.py:293)
            if whole == 0:  # organ-absent → skip (eval.py:295)
                continue
            win_img, win_mask = center_crop_organ(
                image, mask, organ_id, crop_size=crop_size
            )  # (1, 1, d', h', w')
            win_img, win_mask = divisible_pad_end(win_img, win_mask)  # ViT 패치 배수
            organ_feat_dict: dict[str, list] = {}
            m.forward_test_win(
                win_img,  # (1, 1, d', h', w')
                win_mask,  # (1, 1, d', h', w')
                {},  # organ_logits 비움 → 텍스트 대조(484-492) skip
                [organ],  # test_organs (이 장기만)
                {},  # text_feat_dict 미사용
                organ_feat_dict,  # 출력: organ_feat_dict[organ] = image_feat (1,256)
                {organ: whole},  # whole_organ_sizes (완전성 검사용; eval.py:293)
                skip_organ=organ_id,  # 타깃을 경계 검사에서 제외 → 항상 intact
            )
            if organ in organ_feat_dict:  # forward_test_win 이 채웠으면
                feats[organ] = (
                    torch.tensor(organ_feat_dict[organ], device=device)
                    .float()
                    .flatten()
                )  # (256,)
        return feats
