/// GROOPY data contract (v2) — shared by Track A (Recognition) and Track B (Synthesis).
///
/// Recognition EMITS these; Synthesis CONSUMES them. Keep this in lockstep with
/// shared/contract.py and docs/data_contract.md. Bump the version in all three together.
///
/// v2 adds an OPTIONAL [Sentiment] field — additive and backward compatible (v1 payloads
/// without it still parse; it defaults to null).
library groopy.contract;

const String kContractVersion = 'v2';

enum TokenKind { letter, word, control }

/// Sentiment metadata attached to a Token/utterance (v2+). Produced by the sentiment
/// module. What it drives (label vs. signing emphasis vs. expression) is not yet decided.
class Sentiment {
  final String label; // "positive" | "neutral" | "negative"
  final double score; // 0..1

  const Sentiment({required this.label, required this.score});

  factory Sentiment.fromJson(Map<String, dynamic> json) => Sentiment(
        label: json['label'] as String,
        score: (json['score'] as num).toDouble(),
      );

  Map<String, dynamic> toJson() => {'label': label, 'score': score};
}

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

  /// Optional sentiment metadata (v2+). Null when not analysed.
  final Sentiment? sentiment;

  const Token({
    required this.token,
    required this.confidence,
    required this.timestamp,
    required this.kind,
    this.sentiment,
  });

  factory Token.fromJson(Map<String, dynamic> json) => Token(
        token: json['token'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        timestamp: json['timestamp'] as int,
        kind: _kindFromString((json['kind'] as String?) ?? 'letter'),
        sentiment: json['sentiment'] == null
            ? null
            : Sentiment.fromJson(json['sentiment'] as Map<String, dynamic>),
      );

  Map<String, dynamic> toJson() => {
        'token': token,
        'confidence': confidence,
        'timestamp': timestamp,
        'kind': _kindToString(kind),
        'sentiment': sentiment?.toJson(),
      };

  bool get isControl => kind == TokenKind.control;

  @override
  String toString() =>
      'Token($token, ${(confidence * 100).toStringAsFixed(0)}%, $kind)';
}
