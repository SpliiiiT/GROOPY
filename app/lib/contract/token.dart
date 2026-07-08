/// GROOPY data contract (v1) — shared by Track A (Recognition) and Track B (Synthesis).
///
/// Recognition EMITS these; Synthesis CONSUMES them. Keep this in lockstep with
/// recognition/src/token_stream.py and docs/data_contract.md. Bump CONTRACT_VERSION
/// in both places together.
library groopy.contract;

const String kContractVersion = 'v1';

enum TokenKind { letter, word, control }

TokenKind _kindFromString(String s) {
  switch (s) {
    case 'word':
      return TokenKind.word;
    case 'control':
      return TokenKind.control;
    case 'letter':
    default:
      return TokenKind.letter;
  }
}

String _kindToString(TokenKind k) => k.toString().split('.').last;

/// The single object that flows between tracks.
class Token {
  /// Lowercase, normalised: an ASL letter ("a") or word gloss ("hello"),
  /// or a control token: "space" | "del" | "nothing".
  final String token;

  /// 0.0–1.0. Only present when at/above the confidence gate (default 0.80).
  final double confidence;

  /// Unix epoch milliseconds at time of prediction.
  final int timestamp;

  final TokenKind kind;

  const Token({
    required this.token,
    required this.confidence,
    required this.timestamp,
    required this.kind,
  });

  factory Token.fromJson(Map<String, dynamic> json) => Token(
        token: json['token'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        timestamp: json['timestamp'] as int,
        kind: _kindFromString((json['kind'] as String?) ?? 'letter'),
      );

  Map<String, dynamic> toJson() => {
        'token': token,
        'confidence': confidence,
        'timestamp': timestamp,
        'kind': _kindToString(kind),
      };

  bool get isControl => kind == TokenKind.control;

  @override
  String toString() =>
      'Token($token, ${(confidence * 100).toStringAsFixed(0)}%, $kind)';
}
