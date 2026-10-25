package com.hexogen.secondlook

import android.content.Context
import android.net.Uri
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Runs the Second Look preprocessing pipeline on-device.
 *
 * The pipeline itself is the research repo's Python code, executed through
 * Chaquopy (see `src/main/python/`): grayscale -> CLAHE -> breast mask ->
 * pectoral removal -> orientation normalization -> resize -> float32 [0, 1].
 * Running the training code verbatim, rather than a hand-port, is the whole
 * point — a port that drifts silently would feed the model inputs it was never
 * trained on.
 */
object MammogramPreprocessor {

    private val bridge: PyObject
        get() = Python.getInstance().getModule("second_look_bridge")

    /** Model input edge length, read from the shared Python config. */
    val inputSize: Int by lazy { bridge.callAttr("input_size").toInt() }

    /**
     * Preprocess the image at [uri] into the model's input tensor.
     *
     * @return a direct [ByteBuffer] holding `inputSize * inputSize` float32
     *   values in [0, 1], laid out for a `[1, size, size, 1]` tensor.
     * @throws java.io.IOException if the image cannot be read.
     * @throws com.chaquo.python.PyException if the pipeline rejects the image
     *   (empty, corrupt, or an unsupported number of channels).
     */
    fun preprocess(context: Context, uri: Uri): ByteBuffer {
        // The pipeline reads from disk with cv2.imread(IMREAD_UNCHANGED) to keep
        // 16-bit sources at full depth, so hand it a real file rather than
        // decoding to a Bitmap here (which would flatten to 8-bit RGBA first).
        val scratch = File.createTempFile("second_look_input", null, context.cacheDir)
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                scratch.outputStream().use(input::copyTo)
            } ?: throw java.io.IOException("Could not open image: $uri")

            val bytes = bridge
                .callAttr("preprocess_file", scratch.absolutePath, exifRotationDegrees(context, uri))
                .toJava(ByteArray::class.java)

            // TFLite requires a direct buffer. Python wrote little-endian
            // float32, which is what nativeOrder() is on every supported ABI.
            return ByteBuffer.allocateDirect(bytes.size)
                .order(ByteOrder.nativeOrder())
                .put(bytes)
                .apply { rewind() }
        } finally {
            scratch.delete()
        }
    }
}
