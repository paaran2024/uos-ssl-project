import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

/// 이미지 변환 결과
class ImageConversionResult {
  final Uint8List outputBytes;
  final int inferenceTimeMs;

  ImageConversionResult({
    required this.outputBytes,
    required this.inferenceTimeMs,
  });
}

/// 이미지 업스케일링 서비스 (flutter_pytorch_lite 사용)
class ImageConverter {
  static const String _modelAsset = 'assets/models/CATANet-L_x2_1280.ptl';
  static const int _inputSize = 1280;
  static const int _upscale = 2;

  static Module? _module;
  static bool _isLoading = false;

  /// 모델 로드 (최초 1회)
  static Future<void> _loadModel() async {
    if (_module != null || _isLoading) return;

    _isLoading = true;
    try {
      // Asset에서 임시 파일로 복사
      final tempDir = await getTemporaryDirectory();
      final modelPath = '${tempDir.path}/CATANet-L_x2_1280.ptl';
      final modelFile = File(modelPath);

      if (!modelFile.existsSync()) {
        final byteData = await rootBundle.load(_modelAsset);
        await modelFile.writeAsBytes(byteData.buffer.asUint8List());
        debugPrint('모델 파일 복사 완료: $modelPath');
      }

      // 모델 로드
      _module = await FlutterPytorchLite.load(modelPath);
      debugPrint('PyTorch 모델 로드 완료');
    } catch (e) {
      debugPrint('모델 로드 실패: $e');
      rethrow;
    } finally {
      _isLoading = false;
    }
  }

  /// 이미지를 업스케일링합니다.
  static Future<ImageConversionResult> convert(Uint8List inputBytes) async {
    final stopwatch = Stopwatch()..start();

    // 모델 로드
    await _loadModel();
    if (_module == null) {
      throw Exception('모델을 로드할 수 없습니다');
    }

    // 입력 이미지 디코딩
    final inputImage = img.decodeImage(inputBytes);
    if (inputImage == null) {
      throw Exception('이미지를 디코딩할 수 없습니다');
    }

    debugPrint('입력 이미지: ${inputImage.width}x${inputImage.height}');

    // 이미지 리사이즈 (모델 입력 크기에 맞춤)
    final resizedImage = img.copyResize(
      inputImage,
      width: _inputSize,
      height: _inputSize,
      interpolation: img.Interpolation.linear,
    );

    // 이미지를 Float32 텐서로 변환 (CHW 형식, 정규화 0-1)
    final inputData = _imageToTensor(resizedImage);
    final inputShape = Int64List.fromList([1, 3, _inputSize, _inputSize]);
    final inputTensor = Tensor.fromBlobFloat32(inputData, inputShape);

    // 추론 실행
    debugPrint('추론 시작...');
    final input = IValue.from(inputTensor);
    final output = await _module!.forward([input]);
    final outputTensor = output.toTensor();
    debugPrint('추론 완료');

    // 출력 텐서를 이미지로 변환
    final outputData = outputTensor.dataAsFloat32List;
    final outputSize = _inputSize * _upscale;
    final outputImage = _tensorToImage(outputData, outputSize, outputSize);

    // PNG로 인코딩
    final outputBytes = Uint8List.fromList(img.encodePng(outputImage));

    stopwatch.stop();
    debugPrint('업스케일링 완료: ${stopwatch.elapsedMilliseconds}ms');

    return ImageConversionResult(
      outputBytes: outputBytes,
      inferenceTimeMs: stopwatch.elapsedMilliseconds,
    );
  }

  /// 이미지를 Float32 텐서 데이터로 변환 (CHW 형식)
  static Float32List _imageToTensor(img.Image image) {
    final width = image.width;
    final height = image.height;
    final data = Float32List(3 * height * width);

    int idx = 0;
    // CHW 형식: Channel first
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final pixel = image.getPixel(x, y);
          double value;
          switch (c) {
            case 0:
              value = pixel.r / 255.0;
              break;
            case 1:
              value = pixel.g / 255.0;
              break;
            case 2:
              value = pixel.b / 255.0;
              break;
            default:
              value = 0;
          }
          data[idx++] = value;
        }
      }
    }

    return data;
  }

  /// Float32 텐서 데이터를 이미지로 변환 (CHW 형식)
  static img.Image _tensorToImage(Float32List data, int width, int height) {
    final image = img.Image(width: width, height: height);
    final channelSize = height * width;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final idx = y * width + x;
        final r = (data[idx].clamp(0, 1) * 255).round();
        final g = (data[channelSize + idx].clamp(0, 1) * 255).round();
        final b = (data[2 * channelSize + idx].clamp(0, 1) * 255).round();
        image.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return image;
  }

  /// 리소스 해제
  static Future<void> dispose() async {
    await _module?.destroy();
    _module = null;
  }
}
