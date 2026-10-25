#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface MammogramPreprocessor : NSObject
/// Ports data_pipeline/preprocessor.py: grayscale → CLAHE → breast mask
/// → pectoral removal → orientation flip → resize 224 → float32 [0,1].
/// Returns data for a [1, size, size, 1] input tensor. Nil on failure.
+ (nullable NSData *)preprocess:(UIImage *)image size:(int)size;
@end

NS_ASSUME_NONNULL_END
