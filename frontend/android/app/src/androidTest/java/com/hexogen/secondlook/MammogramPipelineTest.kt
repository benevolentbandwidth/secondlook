package com.hexogen.secondlook

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * End-to-end checks for the on-device path: Python preprocessing through
 * TensorFlow Lite inference. Instrumented rather than local because both the
 * CPython runtime and the model live in the APK.
 */
@RunWith(AndroidJUnit4::class)
class MammogramPipelineTest {

    private val context: Context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun inputSizeComesFromTheSharedPythonConfig() {
        assertEquals(224, MammogramPreprocessor.inputSize)
    }

    @Test
    fun preprocessingProducesTheModelInputTensor() {
        val size = MammogramPreprocessor.inputSize
        val tensor = MammogramPreprocessor.preprocess(context, syntheticMammogram())

        assertEquals("float32 [1, $size, $size, 1]", size * size * 4, tensor.capacity())

        val floats = tensor.asFloatBuffer()
        var sum = 0f
        for (i in 0 until floats.limit()) {
            val value = floats.get(i)
            assertTrue("value $value at $i is outside [0, 1]", value in 0f..1f)
            sum += value
        }
        // The pipeline masks background to zero but must not zero everything:
        // an all-black tensor means masking ate the breast region.
        assertTrue("preprocessed tensor is entirely zero", sum > 0f)
    }

    @Test
    fun classificationReturnsAProbabilityAndMatchingTier() {
        val result = MammogramClassifier(context).use { it.classify(syntheticMammogram()) }

        assertTrue("probability ${result.probability} outside [0, 1]", result.probability in 0f..1f)
        assertEquals(MammogramClassifier.tierFor(result.probability), result.tier)
    }

    @Test
    fun tierCutPointsMatchTheSharedThresholds() {
        // Mirrors data_pipeline.label_mapper.TIER_THRESHOLDS (provisional).
        assertEquals("Low", MammogramClassifier.tierFor(0f))
        assertEquals("Low", MammogramClassifier.tierFor(0.329f))
        assertEquals("Moderate", MammogramClassifier.tierFor(0.33f))
        assertEquals("Moderate", MammogramClassifier.tierFor(0.659f))
        assertEquals("Elevated", MammogramClassifier.tierFor(0.66f))
        assertEquals("Elevated", MammogramClassifier.tierFor(1f))
    }

    /**
     * A stand-in mammogram: a bright half-ellipse of "tissue" against a dark
     * background, plus a wedge in the top corner for the pectoral heuristic to
     * find. Enough structure for every stage of the pipeline to do real work.
     */
    private fun syntheticMammogram(): Uri {
        val bitmap = Bitmap.createBitmap(400, 600, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(8, 8, 8))

        val tissue = Paint().apply { color = Color.rgb(190, 190, 190); isAntiAlias = true }
        canvas.drawOval(RectF(-160f, 60f, 280f, 540f), tissue)

        val pectoral = Paint().apply { color = Color.rgb(245, 245, 245) }
        canvas.drawRect(RectF(0f, 0f, 150f, 40f), pectoral)

        val file = File.createTempFile("test_mammogram", ".png", context.cacheDir)
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return Uri.fromFile(file)
    }
}
