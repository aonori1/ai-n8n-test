# このスキルの目的
- Claude Codeが自動売買システムを一貫・安定して実装できるよう、設計方針・優先順位・必須項目を明確化する。
- 毎回の実装で方針がぶれないようにし、品質・安全性・再現性を担保する。

# Claude Codeが従うべき開発方針
- 安全第一：常にリスク管理・サニティチェックを最優先で実装する。
- 小さく分割して確実に動かす：機能は小さなモジュールに分け、単位でテストとレビューを通す。
- 再現性を確保する：バックテストと本番の差分を最小化する。設定（手数料、スリッページ、時間同期など）は明示的に管理する。
- 変更は段階的に：ロールアウトはローカル→ステージング（シミュレーション）→カナリア→本番の順で行う。
- 可観測性を確保：監視・ログ・メトリクスを必須で実装する。可視化がない変更は許可しない。
- 不変性の尊重：秘密情報はコードに埋め込まない。設定は外部管理（環境変数/シークレットマネージャ）。
- 依存関係は固定する：CIでバージョン固定・脆弱性スキャンを必須化。

# 優先するディレクトリ構成
- src/ (実行可能コード)
  - src/cli.py（起動・引数）
  - src/config.py（設定読み込み・検証）
  - src/data/ （マーケットデータ取得・保存）
  - src/models/ （戦略ロジック・フィルタ）
  - src/execution/ （注文発行ラッパー・ブローカーAPI）
  - src/risk/ （リスク管理・ポジション制御）
  - src/backtest/ （バックテストエンジン）
  - src/utils/ （共通ユーティリティ・時間管理・シリアライズ）
  - src/logging/ （ログフォーマット・出力）
- tests/（ユニット・統合テスト）
- infra/（デプロイ・監視設定・Dockerfile）
- docs/（設計・運用手順）
- scripts/（DBマイグレーション、データ準備）
- .ci/（CIワークフロー定義）
- config/（テンプレート設定、staging/prodなど）

# 推奨モジュール分割（役割と公開関数の例）
- config
  - load_config(path) -> validated config object
  - validate_config(config) -> raises on invalid
- market_data
  - fetch_snapshot(symbol, ts) / subscribe_stream(symbols)
  - backtest_data_loader(range, symbol)
- strategy
  - Strategy (クラス): on_tick(tick) -> list[Signal]
  - indicators (pure functions, deterministic)
- execution
  - ExchangeClient (抽象): place_order(order), cancel(order_id), get_order_status(id)
  - BrokerAdapter（ExchangeClientを具象化）: retry/timeout/ratelimit handling
- risk
  - RiskManager: assess(order, current_positions) -> allow/deny/modify
  - PositionManager: reconcile(trades, orders)
- backtest
  - BacktestEngine: run(strategy, data, config) -> trades, metrics
  - PerformanceAnalyzer: compute(returns, drawdown, sharpe)
- logging/monitoring
  - init_logging(env)
  - audit_log(event_type, payload)
  - metrics: gauge(counter).send()
- util
  - timeutil: now_utc(), ensure_time_sync()
  - math: slippage_model(), commission_model()
- persistence
  - TradeStore: append_trade(), query_trades()
  - SnapshotStore: save_market_snapshot()
- tests
  - unit tests for pure functions
  - integration tests with mocked ExchangeClient
  - end-to-end backtest reproducer

# 変更時に必ず確認する項目
- 単体テストが追加されているか（新規機能は必須）
- 既存のテストがすべて通るか（CIで実行）
- バックテスト結果の再現性が保たれているか（seed、データバージョン）
- リスク管理ルールが影響を受けていないか（制限値・ハードストップ）
- 注文発行周りのAPI契約（ExchangeClientのインターフェース）が破壊されていないか
- ログ／メトリクスに必要な情報（order_id、trade_id、理由コード）が出力されているか
- 設定のデフォルト値に危険なものがないか（e.g. test keys in prod）
- レイテンシ/パフォーマンス影響確認（重要ループの計測）
- セキュリティチェック：シークレットがコミットされていないか
- デプロイ手順とロールバック手順が更新されているか

# 自動売買で絶対に軽視してはいけない項目
- リスク制御（ポジション上限、1注文サイズ上限、累積損失停止）
- 注文の冪等性（同一注文の二重発行を防ぐ）
- 注文確認と約定の突合（オフセット・再送時の二重計上防止）
- 時刻同期（サーバと市場時刻のずれを常時監視）
- 例外処理とフェイルセーフ（ネットワーク障害・APIエラー時の状態遷移）
- 監視・アラート（取引停止、異常スリッページ、接続断）
- データ品質（欠損、遅延、リプレイ攻撃）
- 手数料・スリッページ・流動性の現実的見積り
- 監査証跡（全注文・全約定・設定変更の保存）
- リリース管理（段階的ロールアウトと即時停止手段）

# バックテスト、リスク管理、ログ出力の扱い
- バックテスト
  - 完全に分離：backtestモードは実行時に明示フラグで切替え。実際のExecutionモジュールをモックする。
  - 再現性：ランダムシード、データバージョン、設定ファイルを保存して再現可能にする。
  - コストモデル：手数料・スリッページ・スリッページ分布（Worst/Median）を明示的にモデル化する。
  - バリデーション：アウト・オブ・サンプル検証、ウォークフォワード、クロスバリデーションを必須とする。
- リスク管理
  - ランタイム適用：RiskManagerはExecution直前に必ず実行され、注文の拒否・変更を行う。
  - パラメータ化：許容損失、最大ポジション、最大オープン注文数などは外部設定で変更可能にする。
  - ハードストップ：累積損失閾値到達で即時全注文キャンセル・売切りする「キルスイッチ」を実装。
  - 定期監査：ポジションとトレード履歴の突合を自動バッチで行う。
- ログ出力
  - 構造化ログ（JSON）を標準化。必須フィールド：timestamp, level, component, event_type, correlation_id, payload。
  - 監査ログは永続ストレージへ保存（改ざん防止、保持期間の明示）。
  - 重要イベント（発注、約定、キャンセル、リスク拒否、システムエラー）は必ず記録してリアルタイムでアラート対象にする。
  - メトリクス（注文レイテンシ、約定率、slippage分布）はPrometheus等に出力。

# 本番売買コードで禁止すべきこと
- 秘密情報をコード・リポジトリにハードコーディングすること
- リスクチェックをバイパスするフラグを残すこと（例：DISABLE_RISK=true）
- 手動操作での本番変更をドキュメント化せず行うこと
- テストネット用キー／エンドポイントを本番で使用すること
- ログや監視を止めること（デバッグのためのログ無効化も禁止）
- スレッドやプロセス同期を適当に実装して状態不整合を生むこと
- 重大なエラーで自動再試行を無条件に行い続けること（指数バックオフと上限を必須化）
- 依存をunpinnedなまま本番にすること（バージョン固定必須）
- ad-hocのprintや対話的コマンドで状態を変更すること
- 手動でDBを直接書き換えること（操作はスクリプト化・レビュー必須）

# 実装時のチェックリスト（マージ前・デプロイ前の必須項目）
- [ ] コードスタイル／リンター通過
- [ ] 型チェック（mypy等）通過
- [ ] 単体テストカバレッジが基準値以上（プロジェクト規程に従う）
- [ ] 主要な統合テスト（ExchangeClientモック）をCIで実行
- [ ] バックテストの再現スクリプト・結果を添付（seed・データバージョン含む）
- [ ] リスクマネージャのテストケース（拒否・修正シナリオ）を追加
- [ ] ログに必要なフィールドが含まれていることを確認
- [ ] 監視・アラート（閾値含む）をinfraに追加・更新
- [ ] 設定ファイルのバリデーションがあること（load_config が検証）
- [ ] シークレットはシークレットマネージャに格納、リポジトリに無いことを確認
- [ ] パフォーマンスベンチ（注文発行レイテンシ等）の測定結果を添付
- [ ] デプロイ手順、ロールバック手順をドキュメント化
- [ ] カナリア/ステージングでの煙検査（サンドボックス実行）を経ている
- [ ] 監査ログ保存先と保持ポリシーが設定されている
- [ ] 重大障害時のRunbookが作成・レビュー済み

以上を順守すること。必要な場合は該当箇所に必ずユニットテストと運用手順を追加すること。