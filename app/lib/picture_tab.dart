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
    });

    _decodeImageResolution(bytes, (resolution) {
      if (mounted) setState(() => _inputResolution = resolution);
    });
  }

  Future<void> _convertImage() async {
    if (_inputBytes == null) return;

    final result = await ImageConverter.convert(_inputBytes!);

    setState(() {
      _outputBytes = result.outputBytes;
      _inferenceTime = "${result.inferenceTimeMs} ms";
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
                _buildOutputContainer(),
                const SizedBox(height: 30),
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

  Widget _buildInputContainer() {
    return MediaContainer(
      resolution: _inputResolution,
      onTap: _pickImage,
      child: _inputBytes == null
          ? const UploadPlaceholder(text: "image upload")
          : Image.memory(_inputBytes!, fit: BoxFit.contain),
    );
  }

  Widget _buildOutputContainer() {
    return MediaContainer(
      resolution: _outputResolution,
      child: _outputBytes == null
          ? const SizedBox.shrink()
          : Image.memory(_outputBytes!, fit: BoxFit.contain),
    );
  }

  Widget _buildInferenceTime() {
    return Text(
      "Inference time: $_inferenceTime",
      style: AppConstants.inferenceTimeTextStyle,
    );
  }

  Widget _buildActionButton() {
    if (_outputBytes == null) {
      return ActionButton(
        text: "image upscaling",
        icon: Icons.cached_rounded,
        onPressed: _convertImage,
      );
    }

    return ActionButton(
      text: "image download",
      icon: Icons.download_rounded,
      onPressed: _saveImage,
    );
  }
}
