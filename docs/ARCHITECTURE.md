# ARCHITECTURE.md

## 1. システム概要
自動売買システムは、マーケットデータ収集、トレード戦略実行、注文送信、リスク管理、監視／ログ、バックテストを一貫して運用可能にするシステムである。低遅延かつ確実な発注経路と、検証可能なバックテスト基盤を両立し、運用中の安全停止（kill-switch）や多層リスク制御を備えることで商用運用に耐えることを目的とする。

## 2. アーキテクチャ全体像
- イベント駆動 / メッセージ中心
  - Market Data → Message Bus → Strategy Engine / Backtester / Recorder
  - Strategy Engine → Order Manager → Exchange Executor
  - Order Manager ↔ Risk Manager (同期・非同期)
  - Persistence (TimeSeries DB / RDB) に全イベント・注文履歴を保存
  - Monitoring / Logging / Tracing が全コンポーネントを横断
- 分離点
  - バックテスト基盤は本番と物理的・論理的に分離（データ・設定・ネットワーク）
  - 戦略コードはプラグイン化して独立デプロイ可能
- インフラ想定
  - コンテナ化 (Kubernetes)
  - Message Bus (Kafka / RabbitMQ)
  - Timeseries DB (InfluxDB / ClickHouse) + RDB (Postgres)
  - キャッシュ/ロック (Redis)
  - モニタリング (Prometheus, Grafana), ログ集約 (ELK / Loki)
  - トレーシング (OpenTelemetry)

## 3. 主要コンポーネント
- Market Data Ingestor
  - 取引所API (WebSocket/REST/FIX) からのティック/板/約定取得
  - 正規化レイヤー（共通フォーマット）
- Message Bus
  - イベント配信、永続化、スケーラブルなストリーミング
- Strategy Engine
  - 戦略プラグイン実行環境（同期/非同期両対応）
  - シミュレーション・ライブ共通API
- Order Manager
  - 注文の生成・管理・再試行・ステートマシン
  - 注文IDマッピング、手数料計算、注文明細管理
- Executor（Exchange Adapter）
  - 各取引所向けコネクタ（抽象化インターフェース）
  - 送信/キャンセル/訂正の低レイヤー実装
- Risk Manager
  - プリトレードチェック（資金・ポジション・上限）
  - ポストトレード監視、サーキットブレーカー
- Persistence
  - 時系列DB：マーケットデータ、時系列指標
  - RDB：注文履歴、ポジション、ユーザー／設定
  - Audit Log：不変ログ（監査用）
- Backtester
  - 過去データでの戦略検証、トレードシミュレーション、レポート生成
- Admin / Operator UI & API
  - 設定、手動介入、運用ダッシュボード、緊急停止
- Monitoring & Alerting
  - メトリクス収集、アラートルール、ログトレース
- CI/CD / IaC
  - 戦略・サービスのデプロイパイプライン、構成管理

## 4. データフロー
1. Market Data Ingestor が外部取引所からデータを受信（WebSocket/REST/FIX）
2. データは正規化され、Message Bus のトピックに発行（例: market.ticks, market.orderbook）
3. Strategy Engine と Recorder がトピックを購読
   - Strategy Engine は指標を計算し、シグナルを発生
4. シグナルは Order Manager に送信（イベント）
5. Order Manager が預留チェックとプリトレードRiskチェックを行い、注文を生成
6. 生成された注文は Executor 経由で取引所へ送信
7. 取引所からの注文状態（ACK/Fill/Cancel）は Message Bus を通じて Order Manager と Risk Manager に返される
8. Persistence に全イベント・注文・約定を永続化
9. Monitoring にメトリクスとログを送信、アラート発火時は Operator UI に通知
10. 手動／自動のサーキットブレーカーによる停止処理は Order Manager と Executor に伝播

## 5. モジュール責務
- Market Data Ingestor
  - コネクション管理、再接続、シーケンス整合、遅延測定
- Message Bus
  - 高可用な配信、オフセット管理、バックプレッシャ制御
- Strategy Engine
  - 戦略の実行、状態分離（戦略毎にコンテナ/プロセス分離推奨）
- Order Manager
  - 注文ステート管理、再試行ロジック、IDマッピング、整合性保証
- Executor
  - API呼び出し、エラー分類（再試行可能/不可）、レート制限対応
- Risk Manager
  - 事前閾値チェック、累積損失監視、権限管理、停止命令発行
- Persistence
  - イベントの耐久化、スキーマ管理、バックアップ・リストア方針
- Backtester
  - データ整形、Transaction Cost Model、ラグ・スリッページの注記
- Monitoring
  - メトリクス設計（latency, throughput, PnL, positions）、アラート基準
- Admin UI/API
  - 設定管理、運用操作の記録、権限分離

## 6. 外部APIとの接続方針
- 抽象化レイヤー
  - 取引所ごとに Adapter を実装し、共通インターフェース（sendOrder/cancelOrder/getOrderStatus）を提供
- 接続方式
  - マーケットデータ：WebSocket（優先）＋RESTフォールバック
  - 注文送信：REST or FIX（取引所依存）
- 再接続/リトライ
  - エクスポネンシャルバックオフ、最大リトライ回数設定、恒久失敗時はオペレータ通知
- レート制限
  - Adapter 層で局所レートリミッタを実装（トークンバケット）
- 認証
  - 秘密鍵はシークレット管理（Vault等）で格納、必要最小限の権限
- テスト接続
  - testnet/sandbox を優先利用、実トレードはホワイトリストIP等で制限
- 非同期通知（Webhooks）
  - 受信用エンドポイントは認証・署名検証を実装

## 7. ログ・監視方針
- ロギング
  - 構造化ログ（JSON）、相関ID（trace_id、order_id、strategy_id）を必須
  - ログレベル規約（ERROR/WARN/INFO/DEBUG）、DEBUGは環境変数で有効化
  - 監査用の不変ログ（append-only）を別途保管
- メトリクス
  - Prometheus 収集：latency (ms)、throughput (events/s)、注文成功率、スリッページ、PnL、ポジション量
- トレーシング
  - OpenTelemetry による分散トレース（取引のライフサイクル追跡）
- アラート
  - 重大アラート：接続断、注文失敗率閾値超過、PnL急変、サーバーリソース異常
  - 運用アラート：遅延増大、データ欠損、バックプレッシャ検出
- ダッシュボード
  - 戦略ごとの主要KPI（PnL、勝率、最大ドローダウン、取引数、レイテンシ）
- 保持期間・ログローテーション
  - 運用ログ 90日、監査ログ 7年（法規／監査要件に応じて変更）
- テストと検証
  - 監視ルールはシミュレーションで定期検証

## 8. バックテスト・本番の分離方針
- 物理的・論理的分離
  - バックテスト用環境は本番ネットワーク・メッセージバス・DBと分離
  - 構成・シークレットは別管理
- データ管理
  - 履歴データはバージョン管理し、バックテストは特定バージョンのデータセットを参照
  - 本番のマーケットストリームを録音（recording）し、再生用に整形
- コード実行分離
  - 戦略は同一インターフェースだが、バックテスト用ランナーと本番用ランナーを分ける
- 注文流出防止
  - バックテスト環境は実取引コネクタを持たない／sandbox のみ接続
  - 本番環境ではバックテストAPIの許可を禁止
- コンフィグ差分
  - 明示的なモード切替（ENV=production|backtest）を必須化
- テスト自動化
  - CIでバックテストを回し、結果をプルリクエストに添付する運用

## 9. リスク管理の責務分離
- 多層リスク制御
  - 戦略レベル：strategy 自身が内部的リスクパラメータを持つ（最大注文サイズ, max position）
  - サービスレベル（Order Manager 内プリチェック）：即時のプリトレードチェック（資金、ポジション上限）
  - グローバル Risk Manager：クロス戦略合算リスク、累積損失、リアルタイム裁量停止
  - Execution レベル：取引所固有のルール（取引制限、レート制限）を強制
- 権限と権限分離
  - リスク停止・解除は Role-based Permission による二人以上承認フローを採用可能
- 独立性
  - Risk Manager は可能なら別サービスとして実装し、Order Manager から独立して監査・テスト可能にする
- 自動遮断（Circuit Breaker）
  - 異常閾値で自動的に全戦略 or 個別戦略を停止
- ロギング・監査
  - すべてのリスクアクションは不変ログに記録、オペレータ操作は監査ログとして保管

## 10. 今後拡張しやすい構成方針
- モジュール分離（単一責任）
  - 各コンポーネントを小さく保ち、インターフェース契約（API/イベント仕様）を厳格化
- プラグイン化
  - 戦略・取引所アダプタをプラグインとして動的に追加可能にする
- イベント駆動化
  - 内部結合をイベント経由にすることで水平スケールと機能追加を容易にする
- APIファースト設計
  - 内部/外部 API を明確に定義し、バージョニングポリシーを運用
- インフラとしてのコード（IaC）
  - 環境構築は全てコード化、環境差分を最小限に
- コンテナ／オーケストレーション
  - コンテナ化と Kubernetes を前提としたスケール設計
- テスト戦略
  - ユニット・インテグレーション・契約テスト・シミュレーションを整備
- 設定・シークレット管理
  - 外部化された設定管理（Vault/Config Server）で運用の柔軟性を確保
- 可観測性ファースト
  - すべての拡張に対してメトリクス・ログ・トレースの実装を要求
- データ互換性
  - スキーマ進化を考慮したイベントバージョニング（Avro/Protobuf推奨）
- セキュリティ
  - 最小権限、ネットワーク分離、署名付き操作を基本とする

以上。設計決定は運用フェーズの実測（遅延・スループット・障害ケース）に基づきチューニングし、重大な設計変更は可観測性とテストにより段階的に適用する。