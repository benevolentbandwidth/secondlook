package com.hexogen.secondlook.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SwitchColors
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = SystemBlueDark,
    onPrimary = Color.White,
    background = SystemBackgroundDark,
    onBackground = LabelDark,
    surface = SystemBackgroundDark,
    onSurface = LabelDark,
    // The app's vertical background gradient ends on systemGray6, and cards use
    // secondarySystemBackground — the same value on iOS, so one slot covers both.
    surfaceVariant = SystemGray6Dark,
    onSurfaceVariant = SecondaryLabelDark,
    outlineVariant = SeparatorDark
)

private val LightColorScheme = lightColorScheme(
    primary = SystemBlueLight,
    onPrimary = Color.White,
    background = SystemBackgroundLight,
    onBackground = LabelLight,
    surface = SystemBackgroundLight,
    onSurface = LabelLight,
    surfaceVariant = SystemGray6Light,
    onSurfaceVariant = SecondaryLabelLight,
    outlineVariant = SeparatorLight
)

/**
 * Concern-tier colors: SwiftUI's `.green` / `.orange` / `.red` / `.gray`.
 *
 * Kept outside the Material color scheme because they carry meaning — they must
 * not shift with dynamic color or the user's wallpaper.
 */
object TierColors {
    val low: Color @Composable @ReadOnlyComposable get() = pick(SystemGreenLight, SystemGreenDark)
    val moderate: Color @Composable @ReadOnlyComposable get() = pick(SystemOrangeLight, SystemOrangeDark)
    val elevated: Color @Composable @ReadOnlyComposable get() = pick(SystemRedLight, SystemRedDark)
    val unavailable: Color @Composable @ReadOnlyComposable get() = pick(SystemGrayLight, SystemGrayDark)

    @Composable
    @ReadOnlyComposable
    private fun pick(light: Color, dark: Color) = if (isSystemInDarkTheme()) dark else light
}

/**
 * A SwiftUI `Toggle`: green when on, white thumb, no border — rather than
 * Material's primary-tinted switch.
 */
@Composable
fun iOSSwitchColors(): SwitchColors = SwitchDefaults.colors(
    checkedThumbColor = Color.White,
    checkedTrackColor = TierColors.low,
    checkedBorderColor = Color.Transparent,
    uncheckedThumbColor = Color.White,
    uncheckedTrackColor = if (isSystemInDarkTheme()) SwitchOffTrackDark else SwitchOffTrackLight,
    uncheckedBorderColor = Color.Transparent
)

@Composable
fun SecondLookTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    // No dynamic color: the iOS app has a fixed palette, and this one matches it.
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme,
        typography = Typography,
        content = content
    )
}
