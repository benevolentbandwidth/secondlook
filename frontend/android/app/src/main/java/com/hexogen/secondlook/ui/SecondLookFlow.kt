package com.hexogen.secondlook.ui

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.DrawableRes
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.hexogen.secondlook.MammogramClassifier
import com.hexogen.secondlook.R
import com.hexogen.secondlook.createCaptureUri
import com.hexogen.secondlook.loadPreviewBitmap
import com.hexogen.secondlook.ui.theme.SecondLookTheme
import com.hexogen.secondlook.ui.theme.TierColors
import com.hexogen.secondlook.ui.theme.iOSSwitchColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

private const val TAG = "SecondLook"

// MARK: - Flow state

enum class AppStage { Disclaimer, Capture, Scanning, Results }

// MARK: - Root

@Composable
fun SecondLookApp() {
    var stage by rememberSaveable { mutableStateOf(AppStage.Disclaimer) }
    var selectedImage by rememberSaveable { mutableStateOf<Uri?>(null) }
    var result by remember { mutableStateOf<MammogramClassifier.Result?>(null) }

    Box(
        Modifier
            .fillMaxSize()
            // Calm background for the whole app.
            .background(
                Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.background,
                        MaterialTheme.colorScheme.surfaceVariant
                    )
                )
            )
            .safeDrawingPadding()
    ) {
        Crossfade(targetState = stage, label = "stage") { current ->
            when (current) {
                AppStage.Disclaimer -> DisclaimerGate(onAccept = { stage = AppStage.Capture })

                AppStage.Capture -> CaptureScreen(onImageSelected = { uri ->
                    selectedImage = uri
                    stage = AppStage.Scanning
                })

                AppStage.Scanning -> ScanningScreen(
                    image = selectedImage,
                    onComplete = { classification ->
                        result = classification
                        stage = AppStage.Results
                    }
                )

                AppStage.Results -> ResultsScreen(
                    image = selectedImage,
                    result = result,
                    onDone = {
                        // Back to capture — the disclaimer stays accepted for this session.
                        selectedImage = null
                        result = null
                        stage = AppStage.Capture
                    }
                )
            }
        }
    }
}

// MARK: - 1. Disclaimer gate

@Composable
private fun DisclaimerGate(onAccept: () -> Unit) {
    var hasAgreed by rememberSaveable { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.weight(1f))

        HeroIcon(R.drawable.ic_medical_case)
        Spacer(Modifier.height(24.dp))
        Text(
            text = stringResource(R.string.disclaimer_title),
            style = MaterialTheme.typography.displaySmall,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(16.dp))
        Text(
            text = stringResource(R.string.disclaimer_body),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 8.dp)
        )

        Spacer(Modifier.weight(1f))

        Surface(
            shape = RoundedCornerShape(14.dp),
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier
                .fillMaxWidth()
                .toggleable(
                    value = hasAgreed,
                    role = Role.Switch,
                    onValueChange = { hasAgreed = it }
                )
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = stringResource(R.string.disclaimer_agreement),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f)
                )
                Spacer(Modifier.width(12.dp))
                Switch(
                    checked = hasAgreed,
                    onCheckedChange = null,
                    colors = iOSSwitchColors()
                )
            }
        }

        Spacer(Modifier.height(20.dp))

        PrimaryButton(
            text = stringResource(R.string.disclaimer_continue),
            enabled = hasAgreed,
            onClick = onAccept
        )
    }
}

// MARK: - 2. Upload / capture

@Composable
private fun CaptureScreen(onImageSelected: (Uri) -> Unit) {
    val context = LocalContext.current
    var pendingCapture by remember { mutableStateOf<Uri?>(null) }

    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let(onImageSelected) }

    val takePhoto = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { saved -> if (saved) pendingCapture?.let(onImageSelected) }

    val hasCamera = remember {
        context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.weight(1f))

        HeroIcon(R.drawable.ic_add_photo)
        Spacer(Modifier.height(20.dp))
        Text(
            text = stringResource(R.string.capture_title),
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = stringResource(R.string.capture_body),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        )

        Spacer(Modifier.height(28.dp))

        PrimaryButton(
            text = stringResource(R.string.capture_choose_image),
            icon = R.drawable.ic_photo_library,
            modifier = Modifier.padding(horizontal = 20.dp),
            onClick = {
                pickImage.launch(
                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                )
            }
        )

        Spacer(Modifier.height(12.dp))

        PrimaryButton(
            text = stringResource(R.string.capture_take_photo),
            icon = R.drawable.ic_camera,
            enabled = hasCamera,
            modifier = Modifier.padding(horizontal = 20.dp),
            onClick = {
                val uri = createCaptureUri(context)
                pendingCapture = uri
                takePhoto.launch(uri)
            }
        )

        Spacer(Modifier.weight(1f))
    }
}

// MARK: - 3. Scanning

@Composable
private fun ScanningScreen(
    image: Uri?,
    onComplete: (MammogramClassifier.Result?) -> Unit
) {
    val context = LocalContext.current
    val currentOnComplete by rememberUpdatedState(onComplete)

    LaunchedClassification(image, context.applicationContext, currentOnComplete)

    val pulse = rememberInfiniteTransition(label = "pulse")
    val scale by pulse.animateFloat(
        initialValue = 0.9f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(contentAlignment = Alignment.Center) {
            Box(
                Modifier
                    .size(120.dp)
                    .scale(scale)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.12f))
            )
            Icon(
                painter = painterResource(R.drawable.ic_pulse_search),
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(48.dp)
            )
        }

        Spacer(Modifier.height(24.dp))
        Text(
            text = stringResource(R.string.scanning_title),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )
        Spacer(Modifier.height(10.dp))
        IconLabel(
            icon = R.drawable.ic_lock,
            text = stringResource(R.string.scanning_privacy),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall
        )
    }
}

/**
 * Runs preprocessing + inference off the main thread, exactly once per image.
 *
 * A failure here is not exceptional — an unreadable file or an image the
 * pipeline cannot segment both land here — so it degrades to a null result,
 * which the results screen renders as "Analysis Unavailable".
 */
@Composable
private fun LaunchedClassification(
    image: Uri?,
    context: Context,
    onComplete: (MammogramClassifier.Result?) -> Unit
) {
    LaunchedEffect(image) {
        val outcome = withContext(Dispatchers.Default) {
            runCatching {
                val uri = requireNotNull(image) { "No image was selected" }
                MammogramClassifier(context).use { it.classify(uri) }
            }
        }
        outcome.exceptionOrNull()?.let { Log.e(TAG, "Inference failed", it) }
        onComplete(outcome.getOrNull())
    }
}

// MARK: - 4. Results

@Composable
private fun ResultsScreen(
    image: Uri?,
    result: MammogramClassifier.Result?,
    onDone: () -> Unit
) {
    val context = LocalContext.current
    val preview by produceState<Bitmap?>(initialValue = null, image) {
        value = image?.let { withContext(Dispatchers.IO) { loadPreviewBitmap(context, it) } }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        preview?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = stringResource(R.string.cd_selected_image),
                contentScale = ContentScale.Fit,
                // No fillMaxWidth: letting the image size itself within the
                // height cap keeps the layout bounds on the picture, so the
                // shadow hugs the image instead of an empty letterbox.
                modifier = Modifier
                    .heightIn(max = 320.dp)
                    .shadow(elevation = 10.dp, shape = RoundedCornerShape(16.dp))
                    .clip(RoundedCornerShape(16.dp))
            )
            Spacer(Modifier.height(20.dp))
        }

        TierBadge(result)

        Spacer(Modifier.height(16.dp))

        Surface(
            shape = RoundedCornerShape(14.dp),
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(Modifier.padding(16.dp)) {
                IconLabel(
                    icon = R.drawable.ic_info,
                    text = stringResource(R.string.results_meaning_heading),
                    style = MaterialTheme.typography.titleSmall
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(
                        if (result == null) R.string.results_meaning_error
                        else R.string.results_meaning_body
                    ),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                HorizontalDivider(Modifier.padding(vertical = 14.dp))

                IconLabel(
                    icon = R.drawable.ic_stethoscope,
                    text = stringResource(R.string.results_next_step_heading),
                    style = MaterialTheme.typography.titleSmall
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.results_next_step_body),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        Spacer(Modifier.height(24.dp))

        PrimaryButton(
            text = stringResource(R.string.results_scan_another),
            icon = R.drawable.ic_refresh,
            onClick = onDone
        )
    }
}

@Composable
private fun TierBadge(result: MammogramClassifier.Result?) {
    val tierColor = when (result?.tier) {
        null -> TierColors.unavailable
        "Low" -> TierColors.low
        "Moderate" -> TierColors.moderate
        else -> TierColors.elevated
    }
    val tierIcon = when (result?.tier) {
        null -> R.drawable.ic_help
        "Low" -> R.drawable.ic_shield_check
        "Moderate" -> R.drawable.ic_shield_alert
        else -> R.drawable.ic_warning
    }
    val tierTitle = stringResource(
        when (result?.tier) {
            null -> R.string.results_tier_unavailable
            "Low" -> R.string.results_tier_low
            "Moderate" -> R.string.results_tier_moderate
            else -> R.string.results_tier_elevated
        }
    )

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            // SwiftUI's `tierColor.gradient`: the same hue, lifted slightly at
            // the top and shaded at the bottom.
            .background(
                Brush.verticalGradient(
                    listOf(
                        lerp(tierColor, Color.White, 0.05f),
                        lerp(tierColor, Color.Black, 0.15f)
                    )
                )
            )
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            painter = painterResource(tierIcon),
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(28.dp)
        )
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(
                text = stringResource(R.string.results_tier_label),
                style = MaterialTheme.typography.labelMedium,
                color = Color.White.copy(alpha = 0.8f)
            )
            Text(
                text = tierTitle,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
        }
        result?.let {
            Text(
                text = stringResource(
                    R.string.results_confidence,
                    (it.probability * 100).roundToInt()
                ),
                style = MaterialTheme.typography.labelLarge,
                color = Color.White,
                modifier = Modifier
                    .clip(CircleShape)
                    .background(Color.White.copy(alpha = 0.22f))
                    .padding(horizontal = 10.dp, vertical = 5.dp)
            )
        }
    }
}

// MARK: - Shared pieces

@Composable
private fun HeroIcon(@DrawableRes icon: Int) {
    Box(
        modifier = Modifier
            .size(140.dp)
            .clip(CircleShape)
            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.12f)),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            painter = painterResource(icon),
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(56.dp)
        )
    }
}

@Composable
private fun PrimaryButton(
    text: String,
    modifier: Modifier = Modifier,
    @DrawableRes icon: Int? = null,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = RoundedCornerShape(14.dp),
        modifier = modifier
            .fillMaxWidth()
            .height(52.dp)
    ) {
        icon?.let {
            Icon(painterResource(it), contentDescription = null, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
        }
        Text(text, style = MaterialTheme.typography.titleSmall)
    }
}

@Composable
private fun IconLabel(
    @DrawableRes icon: Int,
    text: String,
    modifier: Modifier = Modifier,
    color: Color = MaterialTheme.colorScheme.onSurface,
    style: TextStyle = MaterialTheme.typography.titleSmall
) {
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Icon(
            painter = painterResource(icon),
            contentDescription = null,
            tint = color,
            modifier = Modifier.size(18.dp)
        )
        Spacer(Modifier.width(8.dp))
        Text(text = text, style = style, color = color, fontWeight = FontWeight.SemiBold)
    }
}

// MARK: - Previews

@Preview(showBackground = true)
@Composable
private fun DisclaimerPreview() {
    SecondLookTheme { DisclaimerGate(onAccept = {}) }
}

@Preview(showBackground = true)
@Composable
private fun ResultsPreview() {
    SecondLookTheme {
        ResultsScreen(
            image = null,
            result = MammogramClassifier.Result(probability = 0.42f, tier = "Moderate"),
            onDone = {}
        )
    }
}
