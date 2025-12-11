import 'dart:async';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

/// 전체화면 비디오 플레이어
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
    widget.controller.play();

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

  void _startHideTimer() {
    _hideTimer?.cancel();
    _hideTimer = Timer(const Duration(seconds: 3), () {
      setState(() => _showControls = false);
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

  void _seekRelative(int seconds) {
    final current = widget.controller.value.position;
    final duration = widget.controller.value.duration;

    Duration target = current + Duration(seconds: seconds);
    if (target < Duration.zero) target = Duration.zero;
    if (target > duration) target = duration;

    widget.controller.seekTo(target);
  }

  void _togglePlayPause() {
    setState(() {
      if (widget.controller.value.isPlaying) {
        widget.controller.pause();
      } else {
        widget.controller.play();
      }
    });
    _startHideTimer();
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
            _buildVideoPlayer(),
            _buildControlsOverlay(),
          ],
        ),
      ),
    );
  }

  Widget _buildVideoPlayer() {
    return Center(
      child: AspectRatio(
        aspectRatio: widget.controller.value.aspectRatio,
        child: VideoPlayer(widget.controller),
      ),
    );
  }

  Widget _buildControlsOverlay() {
    return FadeTransition(
      opacity: _fadeAnimation,
      child: _showControls ? _buildControls() : const SizedBox.shrink(),
    );
  }

  Widget _buildControls() {
    return Stack(
      children: [
        _buildBackButton(),
        _buildPlaybackControls(),
        _buildSeekBar(),
      ],
    );
  }

  Widget _buildBackButton() {
    return Positioned(
      top: 40,
      left: 20,
      child: IconButton(
        icon: const Icon(Icons.arrow_back, color: Colors.white, size: 32),
        onPressed: () => Navigator.pop(context),
      ),
    );
  }

  Widget _buildPlaybackControls() {
    return Center(
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          IconButton(
            icon: const Icon(Icons.replay_10, size: 50, color: Colors.white),
            onPressed: () {
              _seekRelative(-10);
              _startHideTimer();
            },
          ),
          const SizedBox(width: 40),
          IconButton(
            icon: Icon(
              widget.controller.value.isPlaying
                  ? Icons.pause_circle
                  : Icons.play_circle,
              size: 70,
              color: Colors.white,
            ),
            onPressed: _togglePlayPause,
          ),
          const SizedBox(width: 40),
          IconButton(
            icon: const Icon(Icons.forward_10, size: 50, color: Colors.white),
            onPressed: () {
              _seekRelative(10);
              _startHideTimer();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildSeekBar() {
    final pos = widget.controller.value.position;
    final dur = widget.controller.value.duration;

    return Positioned(
      bottom: 20,
      left: 0,
      right: 0,
      child: Slider(
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
    );
  }
}
