import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_pytorch_lite/flutter_pytorch_lite.dart';
import 'package:path_provider/path_provider.dart';

/// 이미지 변환 결과
class ImageConversionResult {
  final Uint8List outputBytes;
  final int inferenceTimeMs;
  final int outputWidth;
  final int outputHeight;

  ImageConversionResult({
    required this.outputBytes,
    required this.inferenceTimeMs,
    required this.outputWidth,
    required this.outputHeight,
  });
}

/// 타일 진행 상황
class TileProgress {
  final int currentTile;
  final int totalTiles;
  final double progress; // 0.0 ~ 1.0
  final Uint8List? partialImage; // 중간 결과 이미지

  TileProgress({
    required this.currentTile,
    required this.totalTiles,
    required this.progress,
    this.partialImage,
  });
}

/// 이미지 업스케일링 서비스 (타일링 방식)
class ImageConverter {
  static const String _modelAsset = 'assets/models/output_64.ptl';
  static const int _modelInputSize = 64;
  static const int _upscale = 2;

  // 타일 설정
  static const int _tileSize = 64;
  static const int _overlap = 8;

  static Module? _module;
  static bool _isLoading = false;

  /// 모델 로드 (최초 1회)
  static Future<void> _loadModel() async {
    if (_module != null || _isLoading) return;

    _isLoading = true;
    try {
      final tempDir = await getTemporaryDirectory();
      final modelPath = '${tempDir.path}/output_64.ptl';
      final modelFile = File(modelPath);

      if (!modelFile.existsSync()) {
        final byteData = await rootBundle.load(_modelAsset);
        await modelFile.writeAsBytes(byteData.buffer.asUint8List());
        debugPrint('[DEBUG] 모델 파일 복사 완료: $modelPath');
      }

      _module = await FlutterPytorchLite.load(modelPath);
      debugPrint('[DEBUG] PyTorch 모델 로드 완료');
    } catch (e, stack) {
      debugPrint('[ERROR] 모델 로드 실패: $e');
      debugPrint('[ERROR] Stack: $stack');
      rethrow;
    } finally {
      _isLoading = false;
    }
  }

  /// 이미지를 타일링 방식으로 업스케일링합니다.
  /// [onProgress] 콜백으로 타일 처리 진행 상황과 중간 결과 이미지를 전달합니다.
  static Future<ImageConversionResult> convert(
    Uint8List inputBytes, {
    void Function(TileProgress)? onProgress,
  }) async {
    final stopwatch = Stopwatch()..start();

    try {
      debugPrint('[DEBUG] === 타일링 변환 시작 ===');

      await _loadModel();
      if (_module == null) {
        throw Exception('모델을 로드할 수 없습니다');
      }

      final codec = await ui.instantiateImageCodec(inputBytes);
      final frame = await codec.getNextFrame();
      final originalImage = frame.image;

      final width = originalImage.width;
      final height = originalImage.height;
      debugPrint('[DEBUG] 입력 이미지: ${width}x$height');

      final outputWidth = width * _upscale;
      final outputHeight = height * _upscale;

      final step = _tileSize - _overlap;
      final tilesX = ((width - _overlap) / step).ceil();
      final tilesY = ((height - _overlap) / step).ceil();
      final totalTiles = tilesX * tilesY;
      debugPrint('[DEBUG] 타일 그리드: ${tilesX}x$tilesY = $totalTiles 타일');

      // 출력 버퍼
      final outputBuffer = Uint8List(outputWidth * outputHeight * 4);
      final weightBuffer = Float32List(outputWidth * outputHeight);

      int tileIndex = 0;
      for (int ty = 0; ty < tilesY; ty++) {
        for (int tx = 0; tx < tilesX; tx++) {
          tileIndex++;

          final srcX = (tx * step).clamp(0, width - 1);
          final srcY = (ty * step).clamp(0, height - 1);
          final srcW = (srcX + _tileSize > width) ? width - srcX : _tileSize;
          final srcH = (srcY + _tileSize > height) ? height - srcY : _tileSize;

          debugPrint('[DEBUG] 타일 $tileIndex/$totalTiles: ($srcX,$srcY) ${srcW}x$srcH');

          final tileResult = await _processTile(
            originalImage,
            srcX, srcY, srcW, srcH,
          );

          _blendTileToOutput(
            tileResult,
            outputBuffer,
            weightBuffer,
            srcX * _upscale,
            srcY * _upscale,
            srcW * _upscale,
            srcH * _upscale,
            outputWidth,
            outputHeight,
          );

          // 중간 결과 이미지 생성 및 콜백
          if (onProgress != null) {
            final partialImage = await _createPartialImage(
              outputBuffer,
              weightBuffer,
              outputWidth,
              outputHeight,
            );

            onProgress(TileProgress(
              currentTile: tileIndex,
              totalTiles: totalTiles,
              progress: tileIndex / totalTiles,
              partialImage: partialImage,
            ));
          }
        }
      }

      originalImage.dispose();

      _normalizeOutput(outputBuffer, weightBuffer);
      final outputBytes = await _bufferToPng(outputBuffer, outputWidth, outputHeight);

      stopwatch.stop();
      debugPrint('[DEBUG] === 변환 완료: ${stopwatch.elapsedMilliseconds}ms ===');

      return ImageConversionResult(
        outputBytes: outputBytes,
        inferenceTimeMs: stopwatch.elapsedMilliseconds,
        outputWidth: outputWidth,
        outputHeight: outputHeight,
      );
    } catch (e, stack) {
      debugPrint('[ERROR] 변환 실패: $e');
      debugPrint('[ERROR] Stack: $stack');
      rethrow;
    }
  }

  /// 중간 결과 이미지 생성 (정규화 적용)
  static Future<Uint8List> _createPartialImage(
    Uint8List outputBuffer,
    Float32List weightBuffer,
    int width,
    int height,
  ) async {
    // 임시 버퍼에 정규화된 결과 복사
    final tempBuffer = Uint8List(outputBuffer.length);
    final pixelCount = weightBuffer.length;

    for (int i = 0; i < pixelCount; i++) {
      final weight = weightBuffer[i];
      final idx = i * 4;

      if (weight > 0) {
        tempBuffer[idx] = (outputBuffer[idx] / weight).round().clamp(0, 255);
        tempBuffer[idx + 1] = (outputBuffer[idx + 1] / weight).round().clamp(0, 255);
        tempBuffer[idx + 2] = (outputBuffer[idx + 2] / weight).round().clamp(0, 255);
        tempBuffer[idx + 3] = 255;
      } else {
        // 아직 처리되지 않은 영역은 어둡게 표시
        tempBuffer[idx] = 30;
        tempBuffer[idx + 1] = 30;
        tempBuffer[idx + 2] = 30;
        tempBuffer[idx + 3] = 255;
      }
    }

    return await _bufferToPng(tempBuffer, width, height);
  }

  /// 단일 타일 처리
  static Future<Uint8List> _processTile(
    ui.Image source,
    int srcX, int srcY, int srcW, int srcH,
  ) async {
    final paddedTile = await _extractAndPadTile(source, srcX, srcY, srcW, srcH);

    final inputTensor = await TensorImageUtils.imageToFloat32Tensor(
      paddedTile,
      width: _modelInputSize,
      height: _modelInputSize,
    );
    paddedTile.dispose();

    final input = IValue.from(inputTensor);
    final output = await _module!.forward([input]);
    final outputTensor = output.toTensor();
    final outputData = outputTensor.dataAsFloat32List;

    final outW = srcW * _upscale;
    final outH = srcH * _upscale;
    final tileOutput = Uint8List(outW * outH * 4);

    final modelOutputSize = _modelInputSize * _upscale;
    final channelSize = modelOutputSize * modelOutputSize;

    for (int y = 0; y < outH; y++) {
      for (int x = 0; x < outW; x++) {
        final srcIdx = y * modelOutputSize + x;
        final dstIdx = (y * outW + x) * 4;

        tileOutput[dstIdx] = (outputData[srcIdx].clamp(0, 1) * 255).round();
        tileOutput[dstIdx + 1] = (outputData[channelSize + srcIdx].clamp(0, 1) * 255).round();
        tileOutput[dstIdx + 2] = (outputData[2 * channelSize + srcIdx].clamp(0, 1) * 255).round();
        tileOutput[dstIdx + 3] = 255;
      }
    }

    return tileOutput;
  }

  /// 타일 추출 및 패딩
  static Future<ui.Image> _extractAndPadTile(
    ui.Image source,
    int srcX, int srcY, int srcW, int srcH,
  ) async {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    canvas.drawRect(
      ui.Rect.fromLTWH(0, 0, _modelInputSize.toDouble(), _modelInputSize.toDouble()),
      ui.Paint()..color = const ui.Color(0xFF000000),
    );

    canvas.drawImageRect(
      source,
      ui.Rect.fromLTWH(srcX.toDouble(), srcY.toDouble(), srcW.toDouble(), srcH.toDouble()),
      ui.Rect.fromLTWH(0, 0, srcW.toDouble(), srcH.toDouble()),
      ui.Paint()..filterQuality = ui.FilterQuality.high,
    );

    final picture = recorder.endRecording();
    return await picture.toImage(_modelInputSize, _modelInputSize);
  }

  /// 타일 결과를 출력 버퍼에 블렌딩
  static void _blendTileToOutput(
    Uint8List tileData,
    Uint8List outputBuffer,
    Float32List weightBuffer,
    int dstX, int dstY,
    int tileW, int tileH,
    int outW, int outH,
  ) {
    final overlapOut = _overlap * _upscale;

    for (int y = 0; y < tileH; y++) {
      for (int x = 0; x < tileW; x++) {
        final outX = dstX + x;
        final outY = dstY + y;

        if (outX >= outW || outY >= outH) continue;

        double weight = 1.0;

        if (x < overlapOut && dstX > 0) {
          weight *= x / overlapOut;
        }
        if (x >= tileW - overlapOut && dstX + tileW < outW) {
          weight *= (tileW - x) / overlapOut;
        }
        if (y < overlapOut && dstY > 0) {
          weight *= y / overlapOut;
        }
        if (y >= tileH - overlapOut && dstY + tileH < outH) {
          weight *= (tileH - y) / overlapOut;
        }

        final tileIdx = (y * tileW + x) * 4;
        final outIdx = (outY * outW + outX) * 4;
        final weightIdx = outY * outW + outX;

        outputBuffer[outIdx] = (outputBuffer[outIdx] + tileData[tileIdx] * weight).round().clamp(0, 255 * 10);
        outputBuffer[outIdx + 1] = (outputBuffer[outIdx + 1] + tileData[tileIdx + 1] * weight).round().clamp(0, 255 * 10);
        outputBuffer[outIdx + 2] = (outputBuffer[outIdx + 2] + tileData[tileIdx + 2] * weight).round().clamp(0, 255 * 10);
        outputBuffer[outIdx + 3] = 255;
        weightBuffer[weightIdx] += weight;
      }
    }
  }

  /// 가중치로 출력 정규화
  static void _normalizeOutput(Uint8List outputBuffer, Float32List weightBuffer) {
    final pixelCount = weightBuffer.length;
    for (int i = 0; i < pixelCount; i++) {
      final weight = weightBuffer[i];
      if (weight > 0) {
        final idx = i * 4;
        outputBuffer[idx] = (outputBuffer[idx] / weight).round().clamp(0, 255);
        outputBuffer[idx + 1] = (outputBuffer[idx + 1] / weight).round().clamp(0, 255);
        outputBuffer[idx + 2] = (outputBuffer[idx + 2] / weight).round().clamp(0, 255);
      }
    }
  }

  /// RGBA 버퍼를 PNG로 변환
  static Future<Uint8List> _bufferToPng(Uint8List buffer, int width, int height) async {
    final immutableBuffer = await ui.ImmutableBuffer.fromUint8List(buffer);
    final descriptor = ui.ImageDescriptor.raw(
      immutableBuffer,
      width: width,
      height: height,
      pixelFormat: ui.PixelFormat.rgba8888,
    );
    final codec = await descriptor.instantiateCodec();
    final frame = await codec.getNextFrame();
    final image = frame.image;

    final pngBytes = await image.toByteData(format: ui.ImageByteFormat.png);
    image.dispose();

    return pngBytes!.buffer.asUint8List();
  }

  /// 리소스 해제
  static Future<void> dispose() async {
    await _module?.destroy();
    _module = null;
  }
}
