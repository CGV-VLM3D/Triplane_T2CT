# U5 / S4 — 정렬이 **실제 대응**에서 오는가: 소규모 과적합 + 대조군

실행: `CUDA_VISIBLE_DEVICES=3 python -m tests.repa_probe.u5_overfit.run --steps 200`
(8 볼륨, batch 4, 실제 `Report2CTModule` + `RepaAligner`(mlp, 16³, spatial norm on), 정렬 항만 학습)
원자료: [results/overfit_controls.json](results/overfit_controls.json) · 그림: [figs/S4_overfit_controls.png](figs/S4_overfit_controls.png)

## 결과

| step | 0 | 25 | 50 | 100 | 150 | 199 |
|---|---|---|---|---|---|---|
| **real** cos | −0.000 | +0.202 | +0.297 | +0.414 | +0.477 | **+0.518** |
| **shuffled** cos | −0.000 | +0.189 | +0.280 | +0.369 | +0.424 | **+0.452** |
| **gaussian** cos | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | **+0.000** |
| real − shuffled | 0 | 0.013 | 0.017 | 0.045 | 0.053 | **0.066** |

- `real` — 볼륨과 짝이 맞는 teacher
- `shuffled` — **매 스텝** 배치 안에서 teacher를 새로 뒤섞음 (고정 순열이면 8볼륨에서 매핑을 외워버린다)
- `gaussian` — teacher를 같은 평균/표준편차의 가우시안으로 대체

## 통과한 것

파이프라인은 실제로 정렬을 학습한다. `gaussian`이 200스텝 내내 cos를 **정확히 0**에 유지하는 것이
결정적이다 — 구조가 없는 타깃에는 절대 맞춰지지 않으므로, 올라가는 cosine이 최적화 아티팩트가 아니다.

## ⚠ 그런데 shuffled가 0.452까지 올라간다 — cosine의 대부분은 **위치 prior**다

흉부 CT는 해부학적으로 정형적이다. 토큰 위치 (h, w, d)에 오는 조직이 환자가 달라도 대체로 같으므로,
**"평균적인 teacher 격자"만 예측해도 높은 cosine이 나온다.** 볼륨 고유의 내용에서 오는 몫은
`real − shuffled` = **+0.066**뿐이다(200스텝 기준, 단조 증가 중: 0.013 → 0.045 → 0.066).

이건 REPA에 치명적이지 않다 — 해부학적 구조 prior를 주입하는 것 자체가 이득일 수 있다. 하지만
**raw cosine을 정렬 지표로 읽으면 크게 과대평가하게 된다.** 이 레포는 CLIPScore-I2I에서 이미 같은
교훈을 배웠다([[clipscore-i2i-interpretation]]: 매칭 − 무관 차이로 읽어야 한다). 구조가 똑같다.

**→ 조치**: `RepaAligner`가 매 스텝 `repa_cos_shuffled`(배치를 한 칸 roll한 teacher와의 cosine)와
`repa_cos_gap`을 함께 로깅한다. 추가 forward가 필요 없어 사실상 공짜다. **학습 중 봐야 할 값은
`repa_cos`가 아니라 `repa_cos_gap`이다.**

## ⚠ 관계형 손실의 절대값은 arm 간 비교가 불가능하다

`gaussian`의 rel이 0.0075로 `real`의 0.0206보다 **낮다**. iid 가우시안 토큰의 Gram은 거의 항등행렬이라
맞추기가 쉽기 때문이다. 관계형 손실 값은 같은 teacher 안에서 시간에 따른 감소만 의미가 있고,
서로 다른 타깃 사이의 크기 비교에는 쓸 수 없다.

## 남는 질문

`real − shuffled`가 200스텝에서 아직 증가 중이다. 본 학습(수만 스텝)에서 이 격차가 어디까지 벌어지는지가
"REPA가 볼륨 고유 정보를 정말 전달하는가"의 답이고, U7 런의 `train/repa_cos_gap` 곡선이 그 답을 준다.
