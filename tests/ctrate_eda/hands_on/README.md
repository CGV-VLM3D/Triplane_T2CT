# Hands-on CT-RATE 뷰어 노트북

CT-RATE EDA 번들용 인터랙티브 볼륨/리포트 탐색기. 모든 경로가 교체 가능하므로,
뷰어를 아무 볼륨·마스크·리포트에 갖다 붙일 수 있습니다.

## 실행

```bash
cd /workspace          # `src` import가 해석되도록
jupyter lab            # 또는: jupyter notebook
```

노트북을 열고 **첫 셀을 실행**하세요(이 셀이 `%matplotlib widget`을 설정하고
뷰어를 import합니다). 그다음 셀을 위에서 아래로 실행합니다. 그림은 살아있는
ipympl 위젯입니다 — 슬라이더, 드롭다운, 마우스 호버 HU readout은 실행 중인
Jupyter 커널 안에서만 동작합니다(정적으로 렌더된 `.ipynb`에서는 안 됨).

요구사항(이미 설치됨): `ipympl` 0.10, `ipywidgets` 8.1.

## 노트북

| 파일 | 기능 |
|------|--------------|
| `00_explore.ipynb` | 자유 탐색 — `VOL`/`MASK`를 아무 경로로 지정해 보기. |
| `06_voxel.ipynb`   | 5개 번들 그룹(all-zero vs 이상) + no_chest 노트. |
| `07_multimodal.ipynb` | 질환 아틀라스: 각 볼륨을 그 리포트 + 라벨과 나란히. |
| `08_recon.ipynb`   | recon_1 vs recon_2 연동 패널 — 커널/spacing 차이. |

## 뷰어 API (`from viewer import *`)

- `view_volume(path, mask_path=None)` — orientation 드롭다운, 실시간 slice
  슬라이더, HU 호버 readout, window 프리셋(lung/mediastinal/bone/raw),
  선택적 반투명 마스크 오버레이 토글. 아래에 리포트를 자동 출력합니다.
- `view_recon_pair(path1, path2)` — 커널 차이 확인용 두 연동 패널(orientation +
  window 공유, slice 슬라이더는 독립).
- `load_report(volume_name)` — 임의의 볼륨 이름(예: `valid_1000_a_1`)에 대해
  소견(Findings) / 결론(Impression) + 양성 라벨을 출력.

## 윈도우 (HU level / width)

| 프리셋 | level | width | 용도 |
|--------|------:|------:|-----|
| lung        | -600 | 1500 | 폐 실질, 결절 |
| mediastinal |   40 |  400 | 연부조직, 물질 |
| bone        |  400 | 1800 | 골격, 금속 |
| raw         |   —  |   —  | [-1000, 1000] HU로 클립 |

## 참고

- `_fixed` NIfTI는 이미 HU가 반영되어 있음 — 뷰어는 rescale slope/intercept를
  **적용하지 않습니다**. `-8192` 비-FOV 센티넬은 공기(air) 바닥값으로 매핑됩니다.
- `.nii.gz` 하나 = reconstruction 하나. 리포트와 18개 이상소견 라벨은 scan
  단위입니다(한 scan의 reconstruction들이 공유).
- 데이터셋은 read-only이며, 뷰어는 절대 데이터셋에 쓰지 않습니다.
