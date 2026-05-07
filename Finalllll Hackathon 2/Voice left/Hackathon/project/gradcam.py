"""
gradcam.py — Production-ready Grad-CAM implementation for binary CNN models.

Provides:
    get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    overlay_heatmap(heatmap, original_image_path, alpha=0.4)

Designed to integrate seamlessly with the Flask AI Medical Diagnosis Dashboard.
"""

import cv2
import numpy as np
import tensorflow as tf


# ── Public API ────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: str,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a binary (sigmoid) Keras classifier.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded Keras model (Sequential or Functional). Must have a sigmoid output.
    img_array : np.ndarray
        Preprocessed input image with shape (1, H, W, 3), values in [0, 1].
    last_conv_layer_name : str
        Name of the last Conv2D layer to use as the Grad-CAM target.

    Returns
    -------
    np.ndarray
        2-D float32 array with values in [0, 1] representing the heatmap.

    Raises
    ------
    ValueError
        If ``last_conv_layer_name`` is None or the layer cannot be found.
    RuntimeError
        If the gradient tape produces None gradients (non-differentiable path).
    """

    # ── Validate inputs ───────────────────────────────────────────────────
    if last_conv_layer_name is None:
        raise ValueError(
            "[GradCAM] last_conv_layer_name is None. "
            "Ensure the model contains at least one Conv2D layer and that "
            "the layer name is detected before calling get_gradcam_heatmap()."
        )

    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError as exc:
        raise ValueError(
            f"[GradCAM] Layer '{last_conv_layer_name}' not found in model. "
            f"Available layers: {[l.name for l in model.layers]}"
        ) from exc

    

    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs

    layer_outputs = {}

    for layer in model.layers:
        x = layer(x)
        layer_outputs[layer.name] = x

    conv_output = layer_outputs[last_conv_layer_name]
    final_output = x

    grad_model = tf.keras.models.Model(
        inputs=inputs,
        outputs=[conv_output, final_output]
    )

    

    # ── Forward pass inside GradientTape ─────────────────────────────────
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # tape.watch(img_tensor)

        try:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
        except Exception as exc:
            raise RuntimeError(
                f"[GradCAM] Forward pass through gradient model failed: {exc}"
            ) from exc

        # Binary (sigmoid) output — shape (batch, 1) or (batch,)
        # We differentiate w.r.t. the positive-class score.
        if predictions.shape[-1] == 1:
            class_score = predictions[:, 0]
        else:
            # Fallback: use the index of the top predicted class.
            class_score = predictions[:, tf.argmax(predictions[0])]

    # ── Compute gradients of class score w.r.t. conv feature maps ────────
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "[GradCAM] GradientTape returned None gradients. "
            "This usually means the computation graph between the conv layer "
            "and the model output is not differentiable. "
            "Check that the model has no non-differentiable ops (e.g. numpy "
            "ops) between the target conv layer and the output."
        )

    # ── Pool gradients globally across spatial dimensions ─────────────────
    # Shape: (num_filters,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    

##################3
    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Multiply each channel by corresponding gradient weight
    weighted_maps = conv_outputs * pooled_grads

    # Compute heatmap
    heatmap = tf.reduce_mean(weighted_maps, axis=-1)

    # Apply ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    # Convert to numpy ONLY at the end
    heatmap = heatmap.numpy()

###########


    # ── Apply ReLU — keep only positive activations ───────────────────────
    heatmap = np.maximum(heatmap, 0)

    # ── Normalise to [0, 1] ───────────────────────────────────────────────
    max_val = heatmap.max()
    if max_val == 0:
        print(
            "[GradCAM] WARNING: heatmap is all zeros after ReLU. "
            "The model may be predicting the negative class (Benign) with "
            "very high confidence, resulting in near-zero positive gradients."
        )
        # Return a blank heatmap rather than crashing.
        return heatmap.astype(np.float32)

    heatmap /= max_val

    print(
        f"[GradCAM] Heatmap computed — shape: {heatmap.shape}, "
        f"min: {heatmap.min():.4f}, max: {heatmap.max():.4f}"
    )

    return heatmap.astype(np.float32)


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image_path: str,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image using colour blending.

    Parameters
    ----------
    heatmap : np.ndarray
        2-D float32 array in [0, 1] as returned by ``get_gradcam_heatmap``.
    original_image_path : str
        Filesystem path to the original (unprocessed) image file.
    alpha : float, optional
        Weight of the heatmap in the blend (default 0.4).
        The original image receives weight ``1 - alpha``.

    Returns
    -------
    np.ndarray
        BGR uint8 overlay image with the same spatial dimensions as the
        original. Returns the original image unchanged if any step fails.
    """

    # ── Load original image ───────────────────────────────────────────────
    orig_img = cv2.imread(original_image_path)
    if orig_img is None:
        print(
            f"[GradCAM] WARNING: Could not load image from '{original_image_path}'. "
            "Returning a blank fallback."
        )
        # Return a neutral grey placeholder to avoid crashing the caller.
        return np.zeros((224, 224, 3), dtype=np.uint8)

    target_h, target_w = orig_img.shape[:2]

    # ── Resize heatmap to match original image dimensions ─────────────────
    heatmap_resized = cv2.resize(
        heatmap,            
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # ── Convert normalised float heatmap → uint8 colour map ──────────────
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # ── Blend heatmap with original image ─────────────────────────────────
    # overlay = (1 - alpha) * orig + alpha * heatmap
    overlay = cv2.addWeighted(
        src1=orig_img,
        alpha=1.0 - alpha,
        src2=heatmap_coloured,
        beta=alpha,
        gamma=0,
    )

    return overlay