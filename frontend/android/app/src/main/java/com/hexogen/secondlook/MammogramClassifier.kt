package com.hexogen.secondlook

import android.content.Context
import android.net.Uri
import com.chaquo.python.Python
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * On-device mammogram classifier: preprocess with the Python pipeline, then run
 * the bundled TensorFlow Lite model. Nothing leaves the device.
 *
 * Not thread-safe — a TFLite [Interpreter] must be used from one thread at a
 * time. Create one per scan and [close] it.
 */
class MammogramClassifier(context: Context) : AutoCloseable {

    private val appContext = context.applicationContext
    private val interpreter = Interpreter(loadModel(appContext))

    /** @property probability P(worth a second look). */
    data class Result(val probability: Float, val tier: String)

    fun classify(uri: Uri): Result {
        val input = MammogramPreprocessor.preprocess(appContext, uri)

        val expectedBytes = interpreter.getInputTensor(0).numBytes()
        require(input.capacity() == expectedBytes) {
            "Preprocessed input is ${input.capacity()} bytes, model expects $expectedBytes"
        }

        val outputTensor = interpreter.getOutputTensor(0)
        val output = ByteBuffer.allocateDirect(outputTensor.numBytes())
            .order(ByteOrder.nativeOrder())
        interpreter.run(input, output)
        output.rewind()

        // Single sigmoid output. Clamp before tiering: the tier function
        // rejects values outside [0, 1], and float arithmetic can land a
        // hair beyond either end.
        val probability = output.asFloatBuffer().get(0).coerceIn(0f, 1f)
        return Result(probability, tierFor(probability))
    }

    override fun close() = interpreter.close()

    companion object {
        private const val MODEL_ASSET = "second_look.tflite"

        /**
         * Concern tier for a probability: "Low", "Moderate" or "Elevated".
         *
         * Delegates to the shared Python cut-points rather than duplicating
         * them here — they are provisional and uncalibrated, and will move.
         */
        fun tierFor(probability: Float): String =
            Python.getInstance()
                .getModule("second_look_bridge")
                .callAttr("tier_for", probability)
                .toString()

        /** Memory-maps the model straight out of the APK (see `noCompress`). */
        private fun loadModel(context: Context): MappedByteBuffer =
            context.assets.openFd(MODEL_ASSET).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).use { stream ->
                    stream.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        descriptor.startOffset,
                        descriptor.declaredLength
                    )
                }
            }
    }
}
