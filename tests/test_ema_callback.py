"""EMACallback 단위 테스트.

핵심 계약 4가지를 고정한다.
1. shadow 갱신이 warmup ramp를 포함한 수식과 정확히 일치한다.
2. ``scripts/extract_ema.py`` 추출본의 key 집합이 raw ckpt와 동일하고, lazy 버퍼
   (Report2CTModule의 ``scale_factor``)가 EMA되지 않고 live 값으로 실린다.
3. EMA를 켜도 raw 체크포인트가 비트 단위로 동일하다 (학습 수학 불변).
4. state_dict/load_state_dict 왕복이 shadow를 정확히 복원한다 (resume).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.ema import EMACallback

SCALE_FACTOR = 0.9876  # 첫 배치에서 정해지는 값 (1.0 init과 구분되게)


class _LazyBufferModule(LightningModule):
    """Report2CTModule의 구조적 핵심만 흉내낸 모듈: 파라미터 + lazy 초기화 버퍼."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = torch.nn.Linear(4, 4, bias=False)
        self.register_buffer("scale_factor", torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        if not self._initialized():
            self.scale_factor.fill_(SCALE_FACTOR)  # 첫 배치에서 lazy 초기화
        return self.unet(batch[0]).pow(2).mean()

    def _initialized(self) -> bool:
        return bool(self.scale_factor.item() != 1.0)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _dataloader() -> DataLoader:
    seed_everything(42, workers=True)
    return DataLoader(TensorDataset(torch.randn(16, 4)), batch_size=4)


def _fit(tmp_path, callbacks, max_epochs: int = 4, ckpt_path: str | None = None):
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        default_root_dir=str(tmp_path),
    )
    seed_everything(42, workers=True)
    trainer.fit(_LazyBufferModule(), _dataloader(), ckpt_path=ckpt_path)
    return trainer


def _model_checkpoint(tmp_path) -> ModelCheckpoint:
    """실험 config들과 같은 설정: 전 스냅샷 보존, 2 epoch마다."""
    return ModelCheckpoint(
        dirpath=str(tmp_path / "checkpoints"),
        filename="epoch_{epoch:03d}",
        monitor=None,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=2,
        auto_insert_metric_name=False,
    )


def test_update_matches_closed_form() -> None:
    """warmup ramp d = min(decay, (1+n)/(10+n)) 를 손계산 값과 대조한다."""
    module = _LazyBufferModule()
    with torch.no_grad():
        module.unet.weight.fill_(1.0)
    cb = EMACallback(decay=0.5)
    cb.on_train_start(trainer=None, pl_module=module)  # shadow <- 1.0

    # p=2: n=0, d=min(.5, 1/10)=0.1   -> 1.0*0.1 + 2*0.9 = 1.9
    # p=3: n=1, d=min(.5, 2/11)=2/11  -> 1.9*2/11 + 3*9/11 = 2.8
    # p=4: n=2, d=min(.5, 3/12)=0.25  -> 2.8*0.25 + 4*0.75 = 3.7
    for value, expected in ((2.0, 1.9), (3.0, 2.8), (4.0, 3.7)):
        with torch.no_grad():
            module.unet.weight.fill_(value)
        cb.on_train_batch_end(None, module, None, None, 0)
        got = cb._shadow["unet.weight"]
        assert torch.allclose(got, torch.full_like(got, expected), atol=1e-6), (
            f"p={value}: expected {expected}, got {got.flatten()[0].item()}"
        )
    assert cb._num_updates == 3


def test_extract_ema_matches_raw_keys_and_carries_live_buffer(tmp_path) -> None:
    """추출본은 raw와 같은 key 집합 + live 버퍼 + 다른 가중치를 갖는다."""
    _fit(tmp_path, [_model_checkpoint(tmp_path), EMACallback(decay=0.9)])

    raw_path = tmp_path / "checkpoints" / "epoch_003.ckpt"
    out_path = tmp_path / "derived" / "ema_ep003.ckpt"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/extract_ema.py",
            "--ckpt",
            str(raw_path),
            "--out",
            str(out_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    raw = torch.load(raw_path, weights_only=False)
    ema = torch.load(out_path, weights_only=False)

    # 샘플러(src/eval/samplers/report2ct.py:123)가 읽는 유일한 최상위 키.
    assert list(ema.keys()) == ["state_dict"]
    # strict=True 로드가 통과하려면 key 집합이 완전히 같아야 한다.
    assert set(ema["state_dict"]) == set(raw["state_dict"])
    # 버퍼는 EMA되지 않고 live 값 그대로 (1.0에 얼어붙으면 latent 스케일이 조용히 틀어진다).
    assert (
        ema["state_dict"]["scale_factor"].item()
        == raw["state_dict"]["scale_factor"].item()
        == torch.tensor(SCALE_FACTOR).item()
    )
    # 파라미터는 평균이므로 raw와 달라야 한다.
    assert not torch.equal(
        ema["state_dict"]["unet.weight"], raw["state_dict"]["unet.weight"]
    )


def test_extract_ema_rejects_checkpoint_without_ema(tmp_path) -> None:
    """EMA 없이 학습한 ckpt에 추출을 걸면 조용히 raw를 복사하지 않고 실패한다."""
    _fit(tmp_path, [_model_checkpoint(tmp_path)], max_epochs=2)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/extract_ema.py",
            "--ckpt",
            str(tmp_path / "checkpoints" / "epoch_001.ckpt"),
            "--out",
            str(tmp_path / "derived" / "ema.ckpt"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "no EMA state" in proc.stderr
    assert not (tmp_path / "derived" / "ema.ckpt").exists()


def test_raw_checkpoint_is_bit_identical_with_and_without_ema(tmp_path) -> None:
    """EMA는 순수 additive — 학습 경로에 어떤 영향도 주지 않는다."""
    off = tmp_path / "off"
    on = tmp_path / "on"
    _fit(off, [_model_checkpoint(off)])
    _fit(on, [_model_checkpoint(on), EMACallback(decay=0.9)])

    sd_off = torch.load(off / "checkpoints" / "epoch_003.ckpt", weights_only=False)
    sd_on = torch.load(on / "checkpoints" / "epoch_003.ckpt", weights_only=False)
    assert set(sd_off["state_dict"]) == set(sd_on["state_dict"])
    for key, tensor in sd_off["state_dict"].items():
        assert torch.equal(tensor, sd_on["state_dict"][key]), f"{key} diverged"


def test_state_dict_roundtrip_restores_shadow(tmp_path) -> None:
    """resume 시 Lightning이 복원하는 콜백 상태가 shadow를 정확히 되살린다."""
    ckpt_cb = _model_checkpoint(tmp_path)
    cb = EMACallback(decay=0.9)
    _fit(tmp_path, [ckpt_cb, cb], max_epochs=2)

    saved = torch.load(tmp_path / "checkpoints" / "epoch_001.ckpt", weights_only=False)[
        "callbacks"
    ]["EMACallback"]
    assert saved["num_updates"] == cb._num_updates

    restored = EMACallback(decay=0.9)
    restored.load_state_dict(saved)
    assert restored._num_updates == cb._num_updates
    assert set(restored._shadow) == set(cb._shadow)
    for key, tensor in cb._shadow.items():
        assert torch.equal(restored._shadow[key], tensor.cpu()), f"{key} mismatch"

    # on_train_start는 복원된 shadow를 덮어쓰지 않고 device만 옮긴다.
    restored.on_train_start(trainer=None, pl_module=_LazyBufferModule())
    for key, tensor in cb._shadow.items():
        assert torch.equal(restored._shadow[key], tensor.cpu()), f"{key} clobbered"
