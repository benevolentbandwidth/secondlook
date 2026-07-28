package com.hexogen.secondlook

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import java.io.File

/**
 * Clockwise rotation, in degrees, that the image at [uri] needs to display
 * upright.
 *
 * Camera photos are almost always stored sideways with an EXIF tag, and
 * `cv2.imread(IMREAD_UNCHANGED)` ignores EXIF by design — so the preprocessor
 * has to apply this itself, or the pectoral-muscle and orientation heuristics
 * would run on a rotated breast.
 */
fun exifRotationDegrees(context: Context, uri: Uri): Int {
    val orientation = context.contentResolver.openInputStream(uri)?.use { input ->
        ExifInterface(input).getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )
    } ?: ExifInterface.ORIENTATION_NORMAL

    return when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90
        ExifInterface.ORIENTATION_ROTATE_180 -> 180
        ExifInterface.ORIENTATION_ROTATE_270 -> 270
        else -> 0
    }
}

/**
 * Decode [uri] for display, downsampled so neither edge exceeds [maxDimension]
 * and rotated upright. Returns null if the image cannot be decoded.
 *
 * This is the preview only. The model never sees this bitmap — preprocessing
 * reads the original file so it keeps the source bit depth.
 */
fun loadPreviewBitmap(context: Context, uri: Uri, maxDimension: Int = 1024): Bitmap? {
    // Pass 1: dimensions only. decodeStream always returns null in this mode,
    // so the stream — not the decode result — is what gets null-checked.
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    val stream = context.contentResolver.openInputStream(uri) ?: return null
    stream.use { BitmapFactory.decodeStream(it, null, bounds) }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

    // Pass 2: the real decode, downsampled.
    val options = BitmapFactory.Options().apply {
        inSampleSize = sampleSizeFor(bounds.outWidth, bounds.outHeight, maxDimension)
    }
    val bitmap = context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, options)
    } ?: return null

    val rotation = exifRotationDegrees(context, uri)
    if (rotation == 0) return bitmap

    val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        .also { if (it != bitmap) bitmap.recycle() }
}

/**
 * A content URI the camera app can write a capture into.
 *
 * The file lives in the app's cache — captured mammograms never enter the
 * shared media store, which is the same promise the disclaimer makes.
 */
fun createCaptureUri(context: Context): Uri {
    val captures = File(context.cacheDir, "captures").apply { mkdirs() }
    val file = File.createTempFile("capture_", ".jpg", captures)
    return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
}

private fun sampleSizeFor(width: Int, height: Int, maxDimension: Int): Int {
    var sampleSize = 1
    while (maxOf(width, height) / sampleSize > maxDimension) {
        sampleSize *= 2
    }
    return sampleSize
}
