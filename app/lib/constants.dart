import 'package:flutter/material.dart';

/// 앱 전체에서 사용하는 상수 및 스타일 정의
class AppConstants {
  // 색상
  static const Color primaryColor = Colors.orange;
  static const Color accentColor = Colors.cyan;
  static const Color backgroundColor = Colors.white;
  static Color containerColor = Colors.grey.shade300;
  static Color overlayColor = Colors.black.withValues(alpha: 0.3);

  // 크기
  static const double mediaContainerHeight = 300.0;
  static const double containerBorderRadius = 10.0;
  static const double buttonBorderRadius = 30.0;
  static const double resolutionBadgeRadius = 4.0;

  // 패딩/마진
  static const EdgeInsets containerMargin = EdgeInsets.all(8);
  static const EdgeInsets buttonPadding = EdgeInsets.symmetric(
    horizontal: 24,
    vertical: 12,
  );
  static const EdgeInsets uploadButtonPadding = EdgeInsets.symmetric(
    horizontal: 24,
    vertical: 12,
  );
  static const EdgeInsets resolutionBadgePadding = EdgeInsets.symmetric(
    horizontal: 8,
    vertical: 4,
  );

  // 텍스트 스타일
  static const TextStyle uploadButtonTextStyle = TextStyle(
    color: Colors.white,
    fontWeight: FontWeight.w600,
    fontSize: 16,
  );

  static const TextStyle resolutionTextStyle = TextStyle(
    color: Colors.white,
    fontSize: 12,
  );

  static const TextStyle actionButtonTextStyle = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.bold,
  );

  static const TextStyle inferenceTimeTextStyle = TextStyle(fontSize: 16);
}

/// 공통 위젯 스타일 빌더
class AppStyles {
  /// 미디어 컨테이너 데코레이션
  static BoxDecoration mediaContainerDecoration() {
    return BoxDecoration(
      color: AppConstants.containerColor,
      borderRadius: BorderRadius.circular(AppConstants.containerBorderRadius),
    );
  }

  /// 업로드 버튼 데코레이션
  static BoxDecoration uploadButtonDecoration() {
    return BoxDecoration(
      color: Colors.grey,
      borderRadius: BorderRadius.circular(AppConstants.containerBorderRadius),
    );
  }

  /// 해상도 배지 데코레이션
  static BoxDecoration resolutionBadgeDecoration() {
    return BoxDecoration(
      color: AppConstants.overlayColor,
      borderRadius: BorderRadius.circular(AppConstants.resolutionBadgeRadius),
    );
  }

  /// 액션 버튼 스타일 (upscaling, download 버튼)
  static ButtonStyle actionButtonStyle() {
    return ElevatedButton.styleFrom(
      backgroundColor: AppConstants.primaryColor,
      foregroundColor: Colors.white,
      padding: AppConstants.buttonPadding,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppConstants.buttonBorderRadius),
      ),
    );
  }
}
