import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import ipywidgets as widgets
from matplotlib.widgets import Slider
from scipy.ndimage import zoom


def load_image(image_path):
    """NIfTI 단일 볼륨 파일을 읽어 SimpleITK 이미지로 반환."""
    return sitk.ReadImage(image_path)


def nifti_to_numpy(nifti_image):
    """SimpleITK 이미지를 NumPy 배열 ``(z, y, x)`` 로 변환."""
    return sitk.GetArrayFromImage(nifti_image)


def _match_shape(overlay, target_shape, order):
    """overlay 볼륨을 CT 크기에 맞춰 resize.

    Args:
        overlay: overlay 볼륨 ``(z, y, x)`` (CT와 해상도가 다를 수 있음).
        target_shape: CT 배열의 ``(z, y, x)`` 형상.
        order: 보간 차수 (seg=0 nearest, saliency=1 linear).

    Returns:
        CT와 동일한 ``(z, y, x)`` 로 맞춘 overlay 볼륨.
    """
    if overlay.shape == target_shape:
        return overlay
    factors = [
        t / s for t, s in zip(target_shape, overlay.shape)
    ]  # 축별 배율 (z, y, x)
    return zoom(overlay, factors, order=order)


def _prep_overlay(overlay, target_shape, overlay_type):
    """overlay를 CT 크기에 정합하고 (saliency면) [0,1] 정규화.

    Args:
        overlay: seg 라벨맵 또는 saliency 볼륨 ``(z, y, x)``.
        target_shape: CT 배열의 ``(z, y, x)`` 형상.
        overlay_type: ``'seg'`` 또는 ``'saliency'``.

    Returns:
        CT와 동일한 ``(z, y, x)`` overlay (None이면 그대로 None).
    """
    if overlay is None:
        return None
    order = 0 if overlay_type == "seg" else 1
    overlay = _match_shape(overlay, target_shape, order)  # CT와 동일 (z, y, x)
    if overlay_type == "saliency":
        lo, hi = float(overlay.min()), float(overlay.max())
        overlay = (overlay - lo) / (hi - lo + 1e-8)  # [0, 1] 정규화
    return overlay


def _draw_slice(ax, ct_slice, ov_slice, overlay_type, alpha, saliency_thresh):
    """한 axial 슬라이스(CT + overlay)를 주어진 축에 그림.

    Args:
        ax: 그릴 matplotlib Axes.
        ct_slice: CT 슬라이스 ``(y, x)`` (HU).
        ov_slice: overlay 슬라이스 ``(y, x)`` 또는 None.
        overlay_type: ``'seg'`` 또는 ``'saliency'``.
        alpha: overlay 투명도.
        saliency_thresh: saliency 임계값(정규화 후) 미만은 투명.
    """
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=1000)  # CT (y, x)
    if ov_slice is not None:
        if overlay_type == "seg":
            masked = np.ma.masked_where(ov_slice == 0, ov_slice)  # 배경(0) 투명
            ax.imshow(masked, cmap="tab20", alpha=alpha, interpolation="nearest")
        else:
            masked = np.ma.masked_where(
                ov_slice < saliency_thresh, ov_slice
            )  # 저활성 투명
            ax.imshow(masked, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")


def display_image(
    array, overlay=None, overlay_type="saliency", alpha=0.4, saliency_thresh=0.1
):
    """[Jupyter용] CT 볼륨을 ipywidgets 슬라이더로 스크롤하며 overlay를 겹쳐 표시.

    Args:
        array: CT 볼륨 ``(z, y, x)`` (HU).
        overlay: seg 라벨맵 또는 saliency 볼륨 ``(z, y, x)``; None이면 CT만.
        overlay_type: ``'seg'`` (이산 라벨) 또는 ``'saliency'`` (연속 히트맵).
        alpha: overlay 투명도 (0=안 보임, 1=불투명).
        saliency_thresh: saliency에서 이 값(정규화 후) 미만은 투명 처리.
    """
    overlay = _prep_overlay(overlay, array.shape, overlay_type)

    def view_image(slice_index):
        fig, ax = plt.subplots(figsize=(10, 10))
        ov_slice = None if overlay is None else overlay[slice_index]  # (y, x)
        _draw_slice(
            ax, array[slice_index], ov_slice, overlay_type, alpha, saliency_thresh
        )
        ax.set_title(f"Slice {slice_index}")
        plt.show()

    slice_slider = widgets.IntSlider(
        min=0, max=array.shape[0] - 1, step=1, description="Slice:"
    )
    widgets.interact(view_image, slice_index=slice_slider)


def display_image_window(
    array, overlay=None, overlay_type="saliency", alpha=0.4, saliency_thresh=0.1
):
    """[.py 스크립트용] GUI 창에서 슬라이더 + 마우스 스크롤로 overlay를 겹쳐 표시.

    터미널에서 ``python 3D_volume_viewer.py`` 로 실행. Tk/Qt 같은 GUI backend와
    디스플레이가 필요 (헤드리스 서버에서는 창이 안 뜸 → ``save_slices`` 사용).

    Args:
        array: CT 볼륨 ``(z, y, x)`` (HU).
        overlay: seg 라벨맵 또는 saliency 볼륨 ``(z, y, x)``; None이면 CT만.
        overlay_type: ``'seg'`` 또는 ``'saliency'``.
        alpha: overlay 투명도.
        saliency_thresh: saliency 임계값(정규화 후) 미만은 투명.
    """
    overlay = _prep_overlay(overlay, array.shape, overlay_type)
    n_slices = array.shape[0]
    state = {"idx": n_slices // 2}  # 가운데 슬라이스에서 시작

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.12)  # 하단 슬라이더 공간 확보
    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.04])
    slider = Slider(
        slider_ax, "Slice", 0, n_slices - 1, valinit=state["idx"], valstep=1
    )

    def render(idx):
        ax.clear()
        ov_slice = None if overlay is None else overlay[idx]  # (y, x)
        _draw_slice(ax, array[idx], ov_slice, overlay_type, alpha, saliency_thresh)
        ax.set_title(f"Slice {idx} / {n_slices - 1}")
        fig.canvas.draw_idle()

    def on_slider(val):
        state["idx"] = int(val)
        render(state["idx"])

    def on_scroll(event):
        step = 1 if event.button == "up" else -1
        state["idx"] = int(np.clip(state["idx"] + step, 0, n_slices - 1))
        slider.set_val(state["idx"])  # 슬라이더 갱신 → on_slider가 render 호출

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    render(state["idx"])
    plt.show()


if __name__ == "__main__":
    image_path = ""
    overlay_path = ""  # seg 마스크 또는 saliency 볼륨 (.nii.gz); 없으면 빈 문자열

    ct_array = nifti_to_numpy(load_image(image_path))
    overlay_array = nifti_to_numpy(load_image(overlay_path)) if overlay_path else None

    # .py 스크립트 실행이면 GUI 창 버전 사용 (Jupyter에서는 display_image 호출)
    display_image_window(ct_array, overlay=overlay_array, overlay_type="saliency")
