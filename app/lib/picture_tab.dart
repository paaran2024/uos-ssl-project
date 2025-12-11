import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'constants.dart';
import 'widgets/widgets.dart';
import 'services/services.dart';
import 'media_store_saver.dart';

class PictureTab extends StatefulWidget {
  const PictureTab({super.key});

  @override
  State<PictureTab> createState() => _PictureTabState();
}

class _PictureTabState extends State<PictureTab> {
  Uint8List? _inputBytes;
  Uint8List? _outputBytes;
  String _inferenceTime = "";
  String? _inputResolution;
  String? _outputResolution;

  // 타일링 진행 상황
  bool _isProcessing = false;
  int _currentTile = 0;
  int _totalTiles = 0;
  double _progress = 0.0;
  Uint8List? _partialImage; // 중간 결과 이미지

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);

    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    setState(() {
      _inputBytes = bytes;
      _outputBytes = null;
      _inferenceTime = "";
      _inputResolution = null;
      _outputResolution = null;
      _isProcessing = false;
      _currentTile = 0;
      _totalTiles = 0;
      _progress = 0.0;
      _partialImage = null;
    });

    _decodeImageResolution(bytes, (resolution) {
      if (mounted) setState(() => _inputResolution = resolution);
    });
  }

  Future<void> _convertImage() async {
    if (_inputBytes == null) return;

    setState(() {
      _isProcessing = true;
      _currentTile = 0;
      _totalTiles = 0;
      _progress = 0.0;
      _partialImage = null;
      _outputBytes = null;
    });

    final result = await ImageConverter.convert(
      _inputBytes!,
      onProgress: (progress) {
        if (mounted) {
          setState(() {
            _currentTile = progress.currentTile;
            _totalTiles = progress.totalTiles;
            _progress = progress.progress;
            _partialImage = progress.partialImage;
          });
        }
      },
    );

    setState(() {
      _outputBytes = result.outputBytes;
      _inferenceTime = "${result.inferenceTimeMs} ms";
      _isProcessing = false;
      _partialImage = null;
    });

    _decodeImageResolution(result.outputBytes, (resolution) {
      if (mounted) setState(() => _outputResolution = resolution);
    });
  }

  void _decodeImageResolution(Uint8List bytes, Function(String) onDecoded) {
    ui.decodeImageFromList(bytes, (ui.Image image) {
      onDecoded("${image.width} x ${image.height}");
      image.dispose();
    });
  }

  Future<void> _saveImage() async {
    if (_outputBytes == null) return;

    final ok = await MediaStoreSaver.saveImage(_outputBytes!);
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(ok ? "갤러리에 저장됨!" : "저장 실패")),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: SingleChildScrollView(
            child: Column(
              children: [
                const SizedBox(height: 15),
                _buildInputContainer(),
                if (_isProcessing || _outputBytes != null) _buildOutputContainer(),
                const SizedBox(height: 20),
                _buildInferenceTime(),
                const SizedBox(height: 100),
              ],
            ),
          ),
        ),
        BottomActionButton(child: _buildActionButton()),
      ],
    );
  }

  /// 원본 이미지 (작게 표시)
  Widget _buildInputContainer() {
    return GestureDetector(
      onTap: _pickImage,
      child: Container(
        height: 150,
        width: double.infinity,
        margin: AppConstants.containerMargin,
        decoration: AppStyles.mediaContainerDecoration(),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(AppConstants.containerBorderRadius),
          child: Stack(
            children: [
              SizedBox.expand(
                child: _inputBytes == null
                    ? const UploadPlaceholder(text: "image upload")
                    : Image.memory(_inputBytes!, fit: BoxFit.contain),
              ),
              if (_inputResolution != null)
                Positioned(
                  top: 8,
                  left: 8,
                  child: _buildBadge(_inputResolution!, Colors.blue),
                ),
              if (_inputBytes != null)
                Positioned(
                  top: 8,
                  right: 8,
                  child: _buildBadge("Original", Colors.grey),
                ),
            ],
          ),
        ),
      ),
    );
  }

  /// 출력 이미지 (타일링 진행 상황 또는 최종 결과)
  Widget _buildOutputContainer() {
    // 현재 표시할 이미지 결정
    final displayImage = _outputBytes ?? _partialImage;

    return Container(
      height: 300,
      width: double.infinity,
      margin: AppConstants.containerMargin,
      decoration: AppStyles.mediaContainerDecoration(),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppConstants.containerBorderRadius),
        child: Stack(
          children: [
            // 이미지 표시
            SizedBox.expand(
              child: displayImage != null
                  ? Image.memory(
                      displayImage,
                      fit: BoxFit.contain,
                      gaplessPlayback: true, // 깜빡임 방지
                    )
                  : Center(
                      child: CircularProgressIndicator(
                        color: AppConstants.primaryColor,
                      ),
                    ),
            ),

            // 진행 상황 오버레이 (처리 중일 때만)
            if (_isProcessing)
              Positioned(
                bottom: 0,
                left: 0,
                right: 0,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.bottomCenter,
                      end: Alignment.topCenter,
                      colors: [
                        Colors.black.withValues(alpha: 0.7),
                        Colors.transparent,
                      ],
                    ),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            "Processing tiles...",
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          Text(
                            "$_currentTile / $_totalTiles",
                            style: TextStyle(
                              color: AppConstants.primaryColor,
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(3),
                        child: LinearProgressIndicator(
                          value: _progress,
                          minHeight: 4,
                          backgroundColor: Colors.white24,
                          valueColor: AlwaysStoppedAnimation<Color>(AppConstants.primaryColor),
                        ),
                      ),
                    ],
                  ),
                ),
              ),

            // 해상도 배지 (완료 시)
            if (_outputResolution != null && !_isProcessing)
              Positioned(
                top: 8,
                left: 8,
                child: _buildBadge(_outputResolution!, Colors.green),
              ),

            // x2 Upscaled 라벨 (완료 시)
            if (_outputBytes != null && !_isProcessing)
              Positioned(
                top: 8,
                right: 8,
                child: _buildBadge("x2 Upscaled", AppConstants.primaryColor),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildBadge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.85),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _buildInferenceTime() {
    if (_inferenceTime.isEmpty) return const SizedBox.shrink();

    return Text(
      "Inference time: $_inferenceTime",
      style: AppConstants.inferenceTimeTextStyle,
    );
  }

  Widget _buildActionButton() {
    if (_isProcessing) {
      return ActionButton(
        text: "processing...",
        icon: Icons.hourglass_top,
        onPressed: null,
      );
    }

    if (_outputBytes == null) {
      return ActionButton(
        text: "image upscaling",
        icon: Icons.cached_rounded,
        onPressed: _inputBytes == null ? null : _convertImage,
      );
    }

    return ActionButton(
      text: "image download",
      icon: Icons.download_rounded,
      onPressed: _saveImage,
    );
  }
}
