# U3 — teacher 적합성: 어떤 SPECTRE를, 어느 레이어에서, spatial norm 유무로 쓸 것인가

실행: `CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u3_teacher.run --n-volumes 24`
(2026-07-28, valid_v2 앞 24개 스캔, TF32)
원자료: [results/teacher_spatial.json](results/teacher_spatial.json) · 그림: [figs/U3_cossim_maps.png](figs/U3_cossim_maps.png)

## 판정 (3개 축 모두 결론이 났다)

| 축 | 결정 | 근거 |
|---|---|---|
| **spatial norm** | **ON** | 12개 arm **전부**에서 `_norm`이 `_raw`를 LDS·CDS·RMSC·SRSS 4개 지표 모두에서 이긴다 |
| **layer** | **23 (최종)** | L23 > L17 > L11, 예외 없음. 단 seam도 같이 커진다(아래) |
| **teacher** | **SSL 먼저** (계획대로) | VLA가 명확히 낫지 않다 — LDS는 VLA가 근소 우위, **해부 근거 지표 SRSS와 seam은 SSL 우위** |

## 수치 (몸통 토큰만, n=24 평균, LDS 내림차순)

| arm | LDS↑ | CDS↑ | RMSC↑ | SRSS_lung↑ | seam_within | seam_across | seam_drop↓ | cos_mean |
|---|---|---|---|---|---|---|---|---|
| vla_L23_norm | **0.1039** | 0.0654 | 0.9770 | 0.3926 | 0.7030 | 0.4960 | 0.2941 | 0.0450 |
| **ssl_L23_norm** | 0.0982 | **0.0674** | 0.9727 | **0.4086** | 0.7157 | 0.5519 | 0.2287 | 0.0533 |
| ssl_L23_raw | 0.0899 | 0.0549 | 0.8693 | 0.3481 | 0.8113 | 0.6699 | 0.1743 | 0.2436 |
| vla_L17_norm | 0.0877 | 0.0631 | 0.9718 | 0.3843 | 0.6676 | 0.5363 | 0.1963 | 0.0551 |
| ssl_L17_norm | 0.0857 | 0.0645 | 0.9702 | 0.3928 | 0.6834 | 0.5747 | 0.1588 | 0.0582 |
| vla_L23_raw | 0.0814 | 0.0565 | 0.8991 | 0.3207 | 0.7736 | 0.5982 | 0.2265 | 0.1912 |
| ssl_L17_raw | 0.0764 | 0.0521 | 0.8586 | 0.3266 | 0.7972 | 0.7075 | 0.1126 | 0.2624 |
| vla_L17_raw | 0.0686 | 0.0519 | 0.8677 | 0.3065 | 0.7749 | 0.6691 | 0.1365 | 0.2467 |
| vla_L11_norm | 0.0676 | 0.0506 | 0.9699 | 0.3100 | 0.5649 | 0.5049 | 0.1060 | 0.0588 |
| ssl_L11_norm | 0.0675 | 0.0512 | 0.9658 | 0.3307 | 0.5892 | 0.5405 | 0.0823 | 0.0668 |
| vla_L11_raw | 0.0639 | 0.0428 | 0.8588 | 0.2884 | 0.7240 | 0.6730 | 0.0703 | 0.2621 |
| ssl_L11_raw | 0.0615 | 0.0426 | 0.8403 | 0.2904 | 0.7550 | 0.7086 | 0.0615 | 0.2936 |

## spatial norm이 하는 일이 그림에 그대로 보인다

`_raw` 행은 폐 anchor 하나에 대한 cos-sim이 **화면 전체에서 0.25 근처**다 — 공기든 간이든 척추든
구분이 없다. iREPA가 말한 "patch token의 상당한 global 성분" 그 자체다. `_norm` 행은 배경이 0 이하로
밀려나고 폐만 따뜻하게 남는다. 숫자로도 `cos_mean` 0.244 → 0.053, `RMSC` 0.869 → 0.973,
`SRSS_lung` 0.348 → 0.409 (ssl_L23).

→ **teacher 토큰에 spatial norm을 걸고 정렬한다.** student에는 걸지 않는다 (iREPA Algorithm 1).

## ⚠ crop seam은 실재하고, 깊은 레이어일수록 크다

SPECTRE의 self-attention은 crop(128×128×64 = 토큰 8³) 내부에서만 일어난다. 재조립한 전역 32³
격자에서 **crop 경계를 가로지르는 이웃 토큰의 유사도가 뚝 떨어진다**:

| layer | seam_drop (ssl_norm) |
|---|---|
| 11 | 8.2 % |
| 17 | 15.9 % |
| 23 | **22.9 %** |

우리가 고른 L23이 가장 심하다. 다만 **이게 어디에 영향을 주는지는 손실 종류에 따라 갈린다**:

- **token-wise cosine(hard) 손실은 seam의 영향을 받지 않는다.** student 토큰 각각이 자기 teacher
  토큰과만 맞춰지므로, 이웃 teacher 토큰끼리 불연속인지는 손실에 들어오지 않는다.
- **관계형(Gram / manifold / TRD) 손실만 오염된다.** 게다가 32³ 격자에서 무작위 토큰 쌍의
  **63/64는 애초에 서로 다른 crop**이라, 관계형 신호의 대부분이 "공유 attention 문맥이 없는" 쌍이다.

→ U5의 `RepaAligner`에 **`rel_scope: "global" | "within_crop"` 노브**를 넣는다. `within_crop`은
Gram을 crop(8³ = 512 토큰) 안으로 제한해 seam을 **공짜로** 제거한다(추가 연산 0, 오히려 더 쌈).
비용은 VideoREPA식 cross-slab(z) 장거리 관계를 잃는 것 — 어느 쪽이 나은지는 학습 arm으로 가른다.
overlapping window(비용 4–8×)는 이 두 노브가 모두 실패했을 때의 3순위.

## 방법론 주의

- 지표는 **몸통 토큰만**으로 계산했다(표 전체). CT는 절반이 공기고 공기 토큰끼리는 서로 매우 유사해서
  전체 토큰 기준으로는 모든 지표가 낙관적으로 부풀려진다. 전체 토큰 기준 값도 JSON에 `|all`로 있다.
- iREPA가 보고한 "spatial metric ↔ gFID |r| > 0.85"는 **2D ImageNet 인코더 27개**에서 얻은 값이다.
  3D CT로의 외삽은 가정이고, 이 probe는 "이 teacher에 공간 구조가 있는가 / spatial norm이 그걸 키우는가"를
  같은 축에서 비교하는 용도로만 썼다. 최종 판정은 U7의 실제 FID다.
- SSL vs VLA는 n=24에서 **지표별로 승자가 갈린다**(LDS는 VLA, SRSS·CDS·seam은 SSL). "SSL이 낫다"가
  아니라 "**VLA가 낫다는 증거가 없으니 계획대로 SSL로 시작한다**"가 정확한 표현이다. 두 teacher 모두
  precompute돼 있으므로 재추출 없이 ablation 가능하다.
