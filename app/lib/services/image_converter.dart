import 'package:flutter/foundation.dart';

/// 이미지 변환 결과
class ImageConversionResult {
  final Uint8List outputBytes;
  final int inferenceTimeMs;

  ImageConversionResult({
    required this.outputBytes,
    required this.inferenceTimeMs,
  });
}

/// 이미지 업스케일링 서비스
///
/// TODO: AI 모델을 사용한 업스케일링 로직 구현 필요
class ImageConverter {
  /// 이미지를 업스케일링합니다.
  ///
  /// [inputBytes] 입력 이미지 바이트 (PNG/JPEG)
  /// 반환: 업스케일링된 이미지와 추론 시간
  static Future<ImageConversionResult> convert(Uint8List inputBytes) async {
    // TODO: 실제 AI 모델 추론 로직 구현
    // 현재는 입력을 그대로 반환 (placeholder)

    debugPrint('ImageConverter.convert() - TODO: AI 업스케일링 구현 필요');

    return ImageConversionResult(
      outputBytes: inputBytes,
      inferenceTimeMs: 0,
    );
  }
}
