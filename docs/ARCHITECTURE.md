# docs/ARCHITECTURE.md

# 1 システム概要
- 目的: 市場データをリアルタイムに収集し、複数の自動売買アルゴリズムでシグナル生成→発注→約定管理→決済までを安全に自動化する。
- 対象市場: 仮想通貨・FX・株式など複数の取引所／ブローカーを想定（接続ごとにアダプタ化）。
- 非機能要件（代表値）
  - データ取得遅延目標: <20ms（受信→前処理完了まで）
  - 発注ラウンドトリップ目標: <200ms（受注→約定確認）
  - 可用性目標: 99.9%（ミッションクリティカルコンポーネント）
  - 保持期間: 取引データ（生データ）最低3年、集計メトリクス1年

# 2 アーキテクチャ全体像
- イベント駆動マイクロサービス構成
  - メッセージバス (例: Kafka/RabbitMQ) を中心に非同期連携
  - 各機能は独立デプロイ可能なコンテナサービス（Kubernetes想定）
- レイヤ
  1. データ収集レイヤ（Market Data Ingest）
  2. 前処理・時系列DB（Preprocessor / TSDB）
  3. 戦略エンジン（Strategy Engine）群
  4. リスク管理（Risk Service）
  5. 注文管理（Order Manager / Execution Gateway）
  6. 永続化・分析（Trade DB / Data Warehouse）
  7. 監視・運用（Monitoring / Alerting / Logging）
  8. バックテスト/シミュレーター（隔離環境）

# 3 主要コンポーネント
- Market Data Ingestor
  - WebSocket/REST/FIX対応コネクタ、心拍・再接続・差分処理
- Preprocessor / Normalizer
  - 時刻整合（NTP/PTP）、欠損補間、ティック正規化
- Message Bus
  - トピック設計: market.ticks, market.ohlc, signals, orders, fills, risk.events
- Strategy Engine(s)
  - プラグイン型、状態管理（ステートフル/ステートレス選択可）
- Risk Service
  - リアルタイムのポジション・注文チェック、プリトレードブロッキング
- Order Manager
  - 発注キュー、再試行、順序保証、idempotencyキー
- Execution Gateway
  - 取引所別アダプタ（REST/FIX/WS）、レートリミット管理
- Persistence
  - 時系列DB（InfluxDB/Timescale）＋ RDB（Postgres）＋ W/H（S3）
- Backtester / Simulator
  - 過去データ再生、スリッページ/手数料モデル、戦略検証
- Ops（Monitoring & Logging）
  - Prometheus/Grafana, ELK/Fluentd, Alertmanager
- Secrets & Config
  - Vault等でAPIキー管理、環境別設定

# 4 データフロー
1. 取引所 → Market Data Ingestor（WS/REST/FIX）
2. Ingestor → Preprocessor（時刻整合・正規化）→ メッセージバス(market.ticks)
3. Strategy Engine(s)がmarket.ticksを購読 → シグナル生成 → signalsトピックに送信
4. Risk Serviceがsignalsを購読し、プリトレード検証（静的/動的ルール）
5. 合格したシグナル → Order Manager（注文生成、idempotency付与）
6. Order Manager → Execution Gatewayへ投げる（発注）→ 取引所へ送信
7. 取引所からの約定/注文状態をExecution Gatewayが受領 → fills/ordersトピックへ
8. Persistenceがorders/fillsを永続化、PnL・ポジション集計を更新
9. 監視コンポーネントがメトリクス・ログを収集、アラート発報

# 5 モジュール責務
- Market Data Ingestor
  - 接続維持、順序保証、差分配信、サブスクライブ管理
- Preprocessor
  - タイムスタンプ正規化、欠損処理、レート集計（OHLC）
- Message Bus
  - 耐障害性・遅延保証のあるトピック配信、コンシューマグループ管理
- Strategy Engine
  - 入力データ処理、シグナル生成、状態永続化、パラメタ管理
- Risk Service
  - ルール評価（ポジション/エクスポージャ/日次損失など）、ブロック・修正指示
- Order Manager
  - 注文ライフサイクル管理、再試行、トランザクション境界管理
- Execution Gateway
  - 各取引所プロトコルの実装、レート制御、注文確認の整合性確保
- Persistence
  - トランザクションログ（不変）、集計・照会用のDB管理
- Backtester
  - データ再生、レイトレード挙動のモデリング、戦略スコア出力
- Monitoring/Alerting
  - KPI算出、SLA監視、異常検知、通知エスカレーション

# 6 外部APIとの接続方針
- 接続分類
  - リアルタイム（WebSocket/FIX）: Market Data, Order/Fill Streams
  - バッチ/管理（REST）: 注文発行補助、口座情報、履歴取得
- コネクション管理
  - 接続プール、再接続バックオフ（指数バックオフ＋ジッター）
  - レートリミット管理（トークンバケット）、グローバルと接続別で二重管理
- 信頼性
  - メッセージ順序確認、シーケンス番号/チェックサム検証
  - 冗長接続（複数リージョン／複数APIエンドポイント）でフェイルオーバー
- セキュリティ
  - APIキーはVaultで保管、アクセスは最小権限
  - すべてTLS、通信検証と監査ログ
- 接続パラメタの環境分離
  - Sandbox/SimulationとProductionで完全にキー・接続経路を分離
- 時刻同期
  - NTP/PTPでサーバー時刻を整合、タイムスタンプはUTCで一貫

# 7 ログ・監視方針
- ログ
  - 構造化ログ(JSON)、必須フィールド（timestamp, component, level, trace_id, context）
  - 重要イベント（注文発行、約定、リスクブロック、エラー）は永続ログとして別保管
  - ログ保持ポリシーとアクセス制御（監査ログは長期保存）
- メトリクス
  - ビジネスメトリクス: 約定数、スリッページ率、PnL、戦略別シャープレシオ
  - システムメトリクス: レイテンシ分布、エラー率、キュー長、再接続回数
- トレーシング
  - 分散トレーシング（trace_id）で注文ライフサイクル追跡
- アラート
  - 障害アラート（高優先度）: 発注失敗連続、実行遅延、リスク制限到達
  - 監視アラート（通常優先度）: データ欠損、レート制限接近
  - 通知経路: PagerDuty/Slack/メール、エスカレーションルール
- 運用ダッシュボード
  - リアルタイム表示（ポジション、未約定、戦略KPIs）、履歴分析用ダッシュボード

# 8 バックテスト・本番の分離方針
- 環境分離
  - ネットワーク・認証情報・データベースを完全分離した専用環境を用意
- データ分離
  - バックテスト用はスナップショット化した履歴データセットのみ使用。本番DBへ書き込み不可
- 実行分離
  - バックテスト／シミュレーションはシミュレータ環境（仮想Execution Gateway）で実行
- 設定管理
  - コンフィグは環境ごとに厳格に切り替え（feature flagsは明示的に環境スコープを含む）
- ビルド/デプロイ
  - 同一コードベースでもビルドタグで環境を分離（イメージ: prod/ci/backtest）
- 検証ルール
  - バックテストでの主要結果（PnL・ドローダウン・取引頻度）はリリース候補の必須チェック項目

# 9 リスク管理の責務分離
- 原則
  - リスク判断は独立サービスで行い、Strategyからのビジネスロジックと分離
- レイヤ
  - プリトレードリスク（Order Manager送信前にチェック）
  - ポストトレードリスク（約定後の監査・再評価）
  - グローバルガード（強制停止・サーキットブレーカー）
- ルール管理
  - ルールはバージョン管理され、承認ワークフローで変更を行う
- 権限分離
  - リスク設定変更は特定ロールのみ、変更は監査ログに記録
- フェイルセーフ
  - リスクサービス障害時は保守的デフォルトで全注文ブロックまたは手動承認モードへ遷移
- モニタリング
  - リスクイベントは即時通知、エスカレーションパスを明確化

# 10 今後拡張しやすい構成方針
- インターフェース駆動
  - 明確なメッセージ契約（スキーマ定義: Avro/Protobuf）で互換性を確保
- プラグインアーキテクチャ
  - Strategy・Exchangeアダプタはプラグイン化しホットデプロイを可能に
- バージョニング
  - API/メッセージ・コンポーネントはセマンティックバージョンを付与
- コンテナ化・オーケストレーション
  - 各コンポーネントはコンテナで提供、スケール方針を明示
- 設計の疎結合化
  - 非同期イベント駆動を基本とし、同期呼び出しを最小化
- 設定とパラメタの外部化
  - Feature flags / Runtime config による挙動切替でコード変更を減らす
- テスト戦略
  - 単体・統合・エンドツーエンドに加え、カナリアリリースとカバー率の自動測定
- ドキュメントと運用手順
  - API仕様、デプロイ手順、障害対応手順をコードと同時に保守

添付（運用上の短い運用ルール）
- すべての発注はidempotencyキーを必須化
- 重要設定変更は必ずレビュー・承認プロセスを経由
- 緊急停止（kill-switch）は複数手段で実装（API, CLI, UI）し、監査可能にする