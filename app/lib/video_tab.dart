import 'dart:io';
import 'dart:async';
import 'dart:typed_data';
import 'package:app/media_store_saver.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';


class VideoTab extends StatefulWidget {
  const VideoTab({super.key});

  @override
  State<VideoTab> createState() => _VideoTabState();
}

class _VideoTabState extends State<VideoTab> {
  XFile? _inputVideo;
  XFile? _outputVideo;
  String _inferenceTime = "";

  VideoPlayerController? _inputController;
  VideoPlayerController? _outputController;

  // 비디오 선택
  Future<void> _pickVideo() async {
    final picker = ImagePicker();
    final picked = await picker.pickVideo(source: ImageSource.gallery);

    if (picked != null) {
      setState(() {
        _inputVideo = picked;
        _outputVideo = null;
        _inferenceTime = "";
      });

      _inputController?.dispose();
      _inputController = VideoPlayerController.file(File(picked.path))
        ..initialize().then((_) {
          setState(() {});
        });
    }
  }

  // 변환
  Future<void> _convertVideo() async {
    if (_inputVideo == null) return;

    final stopwatch = Stopwatch()..start();

    // AI 모델 처리 (예시)
    await Future.delayed(const Duration(milliseconds: 300));

    // 변환된 비디오 파일은 입력 비디오와 동일하다고 가정
    _outputVideo = _inputVideo;

    stopwatch.stop();
    _inferenceTime = "${stopwatch.elapsedMilliseconds} ms";

    _outputController?.dispose();
    _outputController =
    VideoPlayerController.file(File(_outputVideo!.path))
      ..initialize().then((_) => setState(() {}));

    setState(() {});
  }

  // 모바일 갤러리 저장
  Future<void> _saveOutputVideoToGallery() async {
    if (_outputVideo == null) return;

    try {
      final Uint8List bytes = await _outputVideo!.readAsBytes();

      bool ok = await MediaStoreSaver.saveVideo(bytes);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(ok ? "갤러리에 저장됨" : "저장 실패")),
      );
    } catch (e) {
      print("비디오 저장 중 오류 발생: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("비디오 저장 실패")),
      );
    }
  }

  @override
  void dispose() {
    _inputController?.dispose();
    _outputController?.dispose();
    super.dispose();
  }

  // 전체화면 페이지
  void _openFullScreen(VideoPlayerController controller) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => FullScreenVideoPlayer(controller: controller),
      ),
    );
  }

  // 비디오 플레이어 위젯 생성
  Widget _buildVideoPlayer({
    required String placeholderText,
    required VideoPlayerController? controller,
    required bool isInput,
    required VoidCallback? onTap,
  }) {
    if (controller == null) {
      return GestureDetector(
        onTap: onTap,
        child: Container(
          height: 250,
          width: double.infinity,
          margin: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.grey.shade300,
            borderRadius: BorderRadius.circular(10),
          ),
          child: Center(child: Text(placeholderText)),
        ),
      );
    }

    if (!controller.value.isInitialized) {
      return Container(
        height: 250,
        width: double.infinity,
        margin: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.grey.shade300,
          borderRadius: BorderRadius.circular(10),
        ),
        child: const Center(child: CircularProgressIndicator()),
      );
    }

    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        margin: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: Colors.grey.shade300,
          borderRadius: BorderRadius.circular(10),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: AspectRatio(
            aspectRatio: controller.value.aspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                VideoPlayer(controller),
                // 재생/일시정지 버튼
                Center(
                  child: IconButton(
                    icon: Icon(
                      controller.value.isPlaying
                          ? Icons.pause_circle
                          : Icons.play_circle,
                      size: 40,
                      color: Colors.white,
                    ),
                    onPressed: () {
                      setState(() {
                        if (controller.value.isPlaying) {
                          controller.pause();
                        } else {
                          controller.play();
                        }
                      });
                    },
                  ),
                ),
                // 출력 영상에만 전체화면 버튼 추가
                if (!isInput)
                  Positioned(
                    bottom: 10,
                    right: 10,
                    child: IconButton(
                      icon: const Icon(Icons.fullscreen,
                          size: 36, color: Colors.white),
                      onPressed: () {
                        _openFullScreen(controller);
                      },
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        children: [
          // 입력 비디오
          _buildVideoPlayer(
            placeholderText: "입력 영상 선택",
            controller: _inputController,
            isInput: true,
            onTap: _pickVideo,
          ),

          // 출력 비디오
          _buildVideoPlayer(
            placeholderText: "출력 영상",
            controller: _outputController,
            isInput: false,
            onTap: null,
          ),

          const SizedBox(height: 20),

          // 변환 버튼
          ElevatedButton(
            onPressed: _convertVideo,
            child: const Text("변환"),
          ),

          const SizedBox(height: 16),

          // 갤러리에 저장 버튼
          ElevatedButton(
            onPressed: _outputVideo == null ? null : _saveOutputVideoToGallery,
            child: const Text("갤러리에 저장"),
          ),

          const SizedBox(height: 16),

          Text(
            "inference time: $_inferenceTime",
            style: const TextStyle(fontSize: 16),
          ),

          const SizedBox(height: 20),
        ],
      ),
    );
  }
}

// 전체화면 플레이어 페이지
class FullScreenVideoPlayer extends StatefulWidget {
  final VideoPlayerController controller;

  const FullScreenVideoPlayer({super.key, required this.controller});

  @override
  State<FullScreenVideoPlayer> createState() => _FullScreenVideoPlayerState();
}

class _FullScreenVideoPlayerState extends State<FullScreenVideoPlayer>
    with SingleTickerProviderStateMixin {
  bool _showControls = false;
  Timer? _hideTimer;

  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();

    widget.controller.play(); // 자동 재생

    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );

    _fadeAnimation = Tween<double>(begin: 0, end: 1).animate(_fadeController);

    _startHideTimer();
  }

  @override
  void dispose() {
    _hideTimer?.cancel();
    _fadeController.dispose();
    super.dispose();
  }

  // 3초 후 자동 숨김
  void _startHideTimer() {
    _hideTimer?.cancel();
    _hideTimer = Timer(const Duration(seconds: 3), () {
      setState(() {
        _showControls = false;
      });
      _fadeController.reverse();
    });
  }

  void _toggleControls() {
    setState(() => _showControls = !_showControls);

    if (_showControls) {
      _fadeController.forward();
      _startHideTimer();
    } else {
      _fadeController.reverse();
    }
  }

  // 10초 이동
  void _seekRelative(int seconds) {
    final current = widget.controller.value.position;
    final duration = widget.controller.value.duration;

    Duration target = current + Duration(seconds: seconds);

    if (target < Duration.zero) target = Duration.zero;
    if (target > duration) target = duration;

    widget.controller.seekTo(target);
  }

  // 시크바
  Widget _buildSeekBar() {
    final pos = widget.controller.value.position;
    final dur = widget.controller.value.duration;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Slider(
          min: 0,
          max: dur.inMilliseconds.toDouble(),
          value: pos.inMilliseconds.clamp(0, dur.inMilliseconds).toDouble(),
          onChangeStart: (_) => _hideTimer?.cancel(),
          onChanged: (value) {
            widget.controller.seekTo(Duration(milliseconds: value.toInt()));
            setState(() {});
          },
          onChangeEnd: (_) => _startHideTimer(),
          activeColor: Colors.white,
          inactiveColor: Colors.white70,
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: _toggleControls,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // 비디오 - 화면 비율에 맞춰 표시
            Center(
              child: AspectRatio(
                aspectRatio: widget.controller.value.aspectRatio,
                child: VideoPlayer(widget.controller),
              ),
            ),

            // Fade-in Controls (버튼 + 시크바)
            FadeTransition(
              opacity: _fadeAnimation,
              child: _showControls
                  ? Stack(
                children: [
                  // 전체화면 뒤로가기 버튼
                  Positioned(
                    top: 40,
                    left: 20,
                    child: IconButton(
                      icon: const Icon(Icons.arrow_back,
                          color: Colors.white, size: 32),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ),

                  // 중앙 컨트롤러
                  Center(
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // << 10초 되감기
                        IconButton(
                          icon: const Icon(Icons.replay_10,
                              size: 50, color: Colors.white),
                          onPressed: () {
                            _seekRelative(-10);
                            _startHideTimer();
                          },
                        ),

                        const SizedBox(width: 40),

                        // 재생 / 일시정지
                        IconButton(
                          icon: Icon(
                            widget.controller.value.isPlaying
                                ? Icons.pause_circle
                                : Icons.play_circle,
                            size: 70,
                            color: Colors.white,
                          ),
                          onPressed: () {
                            setState(() {
                              if (widget.controller.value.isPlaying) {
                                widget.controller.pause();
                              } else {
                                widget.controller.play();
                              }
                            });
                            _startHideTimer();
                          },
                        ),

                        const SizedBox(width: 40),

                        // >> 10초 빨리감기
                        IconButton(
                          icon: const Icon(Icons.forward_10,
                              size: 50, color: Colors.white),
                          onPressed: () {
                            _seekRelative(10);
                            _startHideTimer();
                          },
                        ),
                      ],
                    ),
                  ),

                  // 하단 시크바
                  Positioned(
                    bottom: 20,
                    left: 0,
                    right: 0,
                    child: _buildSeekBar(),
                  ),
                ],
              )
                  : const SizedBox.shrink(),
            ),
          ],
        ),
      ),
    );
  }
}