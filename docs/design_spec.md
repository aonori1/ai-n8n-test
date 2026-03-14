以下は、自動売買システムの設計書（Markdown）のテンプレート／完成版です。与えられた要件（{{$json.content}}）に基づいてカスタマイズしてください。要件の追加・変更があれば、その部分を反映して再生成します。

# 1. システム概要
目的:
- 金融市場（例: 仮想通貨、FX、株式）向けの自動売買システムを提供する。
- リアルタイム市場データを取得し、戦略（Strategy）でシグナル生成、リスク管理を行った上で注文を執行する。
- バックテスト/フォワードテスト、モニタリング、運用管理（監査・履歴）をサポートする。

主要機能:
- マーケットデータ取り込み（ティック/板/約定）
- 戦略エンジン（複数戦略の同時稼働）
- 注文管理（OMS）と実行ゲートウェイ（FIX/REST/WebSocket）
- リスク管理（プリトレード、ポストトレード、総括リスク）
- ポジション管理／P&L計算
- バックテスト・シミュレーション基盤
- ログ・監視・アラート
- Web管理UI / API

非機能要件（例）:
- レイテンシ: マーケット→発注のパスはターゲット 10ms〜100ms（戦略特性により変動）
- 可用性: 24/7運用。主要コンポーネントは冗長化。SLA 99.9% 以上を目標
- スケーラビリティ: 戦略数・銘柄数の増加に対応
- セキュリティ: API認証、秘密情報暗号化、操作権限分離

# 2. システムアーキテクチャ
全体構成（論理）:
- Ingress Layer: Market Data Adapter, Broker/Exchange Adapter
- Messaging Layer: Pub/Sub（低遅延メッセージバス）
- Core Services:
  - Market Data Service
  - Strategy Engine Service
  - Risk Engine Service
  - Order Management Service (OMS)
  - Execution Gateway Service
  - Position/P&L Service
  - Backtest/Simulator Service
- Persistence Layer: Time-series DB、トランザクショナルDB、データウェアハウス
- Infra/Platform: Kubernetesクラスタ、CI/CD、Secrets Manager
- Observability: Logging, Metrics, Tracing, Alerting
- UI/API Layer: REST/GraphQL, WebSocket, Web UI, CLI

配置図（簡易フロー）:
Market Data Adapters → Message Bus → Market Data Service → Strategy Engine → Risk Engine → OMS → Execution Gateway → Exchange
Execution Gateway ← Execution Reports ← Exchange → OMS → Position Service → P&L/Reporting

高可用構成:
- 各サービスは複数レプリカで運用、ステートフル部分はDBと専用リーダー/ライター構成
- Execution Gateway はリージョン冗長（必要に応じて）またはアクティブ/スタンバイ
- メッセージバス（Kafka/Redis Streams）はクラスタ構成で耐障害性確保

# 3. コンポーネント構成
1. Market Data Adapter
   - 取引所API (REST/WebSocket/FIX) をラップし、標準化されたティック/板データを配信。
2. Market Data Service
   - データ整形、正規化（timestamp, symbol mapping）、軽度の集計（分足）を行いMessage Busへ配信。履歴はTSDBに保存。
3. Message Bus
   - 高スループット低レイテンシのPub/Sub（Kafka/Redis Streams/NATS等）。
4. Strategy Engine
   - 戦略プラグインをロードしてリアルタイムにシグナル生成。Backtesting Mode と Live Mode を切替可能。
5. Risk Engine
   - プリトレードチェック（注文前: 取引可能額、ポジション制限、取引制限）、ポストトレード監視（実行後リスク集計）、グローバル制御（回転率、エクスポージャ、銘柄別上限）。
6. Order Management Service (OMS)
   - 注文のライフサイクル管理（New, PartiallyFilled, Filled, Cancelled, Rejected）。オーダーブックの保持、再送/リプレイ機能。
7. Execution Gateway
   - 実取引所へ接続。FIX/REST/WebSocket クライアント。注文送信・ステータス受信を行う。
8. Position & P&L Service
   - 約定情報を取り込み、ポジション更新、マークツーマーケット、リアルタイムP&L算出。
9. Backtest/Simulator
   - 履歴データを用いた戦略検証。マイクロサービスタイプで戦略と同APIで稼働可能。
10. UI & API
    - 運用管理画面、ダッシュボード、戦略管理、モニタ、ログ検索API。
11. Persistence
    - Postgres（取引ログ/メタデータ）、ClickHouse（イベント集計/履歴クエリ）、InfluxDB/Timescale（時系列）、S3互換（深履歴/バックアップ）。
12. Observability
    - Prometheus + Grafana, ELK/EFK（Elasticsearch/Fluentd/Kibana）、Jaeger/OpenTelemetry。
13. Auth & Secrets
    - Vault/AWS KMS、OAuth2/OpenID Connect ベースの認証・権限管理。

# 4. データフロー
1. マーケットデータ受信
   - Adapter が取引所から生データを取得 → 正規化 → Message Bus に publish（topic: market.ticks.symbol）
2. 戦略によるシグナル生成
   - Strategy Engine は該当トピックをsubscribe し、インジケータ計算 → シグナル（order request）を生成
3. プリトレードリスクチェック
   - Strategy からの注文要求を Risk Engine が検証（資金・ポジション・取引ルール）→ OK/Reject
4. OMS 経由で発注
   - OMS が ID を付与して Execution Gateway に送信 → 取引所へ送付
5. 実行レポート受取
   - Execution Gateway が約定/キャンセル/拒否を受信 → OMS/Message Bus に反映
6. ポジション更新
   - Position Service が約定イベントを取り込み、ポジション・P&L を更新
7. ログ・監視
   - すべてのイベントはログ（取引ログ, アクセスログ）に記録しメトリクスと合わせて監視/アラート判定
8. バックテスト
   - 履歴データで同じStrategy Engine を動かし、同APIでシミュレーション実行、レポート生成

# 5. 使用技術スタック
推奨（例）:
- 言語
  - 戦略: Python（pandas, numpy, async libs）／または JVM/Go/Rust（低レイテンシ必要時）
  - Core Services: Go / Java / Kotlin / Rust（高信頼性・並列処理向け）
- メッセージング: Kafka / NATS / Redis Streams
- DB: PostgreSQL（トランザクションデータ）、ClickHouse（時系列集計）、TimescaleDB or InfluxDB（ティックデータ）、S3（深履歴）
- Execution: QuickFIX/J or QuickFIX/ (FIX), HTTP/WebSocket clients
- コンテナ/オーケストレーション: Docker + Kubernetes
- CI/CD: GitHub Actions / GitLab CI / Jenkins
- Secrets: HashiCorp Vault / AWS Secrets Manager
- Observability: Prometheus + Grafana, ELK/EFK, OpenTelemetry / Jaeger
- Authentication: Keycloak / OAuth2
- Infrastructure: Terraform for IaaC
- Backtest: vectorized pandas/numpy, zipline-likeフレームワーク or custom simulator

# 6. モジュール構成
サービス単位のモジュール一覧（例）:

- common/
  - config, logging, metrics, auth, error definitions, utils
- adapters/
  - exchange_adapter/
    - exchange_x_adapter.py (WebSocket/REST/FIX wrapper)
    - exchange_y_adapter.py
- market_data/
  - ingester.py, normalizer.py, aggregator.py
- messaging/
  - producer.py, consumer.py, topic_definitions.py
- strategy/
  - strategy_api.py (Strategy interface)
  - strategies/
    - mean_reversion.py
    - momentum.py
- risk/
  - pre_trade.py, post_trade.py, limit_store.py, circuit_breaker.py
- oms/
  - order_controller.py, order_store.py, retry_policy.py
- execution/
  - executor.py, fix_client.py, rest_client.py
- position/
  - position_store.py, pnl_calculator.py
- backtest/
  - backtest_engine.py, scenario_runner.py, report_generator.py
- api/
  - rest_api.py, websocket_api.py, auth_middleware.py
- ui/
  - frontend (React/Vue)
- infra/
  - k8s_manifests/, terraform/
- tests/
  - unit/, integration/, e2e/
- scripts/
  - deploy.sh, migrate_db.sh, start_sim.sh

各モジュールは明確なインターフェース（gRPC/REST）で通信することを推奨。

# 7. ディレクトリ構成
リポジトリトップ例（monorepo想定）:

- /repo-root
  - /services
    - /market-data-service
    - /strategy-service
    - /risk-service
    - /oms-service
    - /execution-service
    - /position-service
    - /backtest-service
  - /libs
    - /common
    - /exchange-clients
    - /strategy-sdk
  - /infra
    - /k8s
    - /terraform
  - /deploy
    - helm-charts, manifests
  - /web-ui
    - frontend
  - /scripts
  - /docs
  - /tests
  - .github/workflows/
  - README.md

各サービスは独立した Dockerfile、CI/CD パイプラインを持つ。

# 8. 外部API設計
APIはREST + WebSocketを提供。内部サービス間はgRPC推奨。

主要エンドポイント例（REST / JSON）:

1) 認証
- POST /api/v1/auth/token
  - req: { "client_id", "client_secret" }
  - resp: { "access_token", "expires_in" }

2) 戦略管理
- GET /api/v1/strategies
- POST /api/v1/strategies
  - req: { "name", "description", "config", "entry_point" }
- PUT /api/v1/strategies/{id}/enable
- PUT /api/v1/strategies/{id}/disable

3) 注文操作
- POST /api/v1/orders
  - req: {
      "client_order_id",
      "symbol",
      "side": "BUY|SELL",
      "type": "MARKET|LIMIT|IOC|FOK",
      "price": float|null,
      "quantity": float,
      "strategy_id": string|null,
      "metadata": {...}
    }
  - resp: { "order_id", "status", "accepted_at" }
- GET /api/v1/orders/{order_id}
  - resp: {order_status, fills[], timestamps, meta}
- DELETE /api/v1/orders/{order_id}  // cancel

4) ポジション・P&L
- GET /api/v1/positions
  - resp: [ { symbol, qty, avg_price, unrealized_pnl, realized_pnl } ]
- GET /api/v1/pnl?from=...&to=...

5) 市場データ・履歴
- GET /api/v1/market/ticks?symbol=X&from=...&to=...
- GET /api/v1/market/ohlc?symbol=X&interval=1m&from=...

6) WebSocket（リアルタイム）
- サブスクリプション: market.ticks, order.updates, position.updates, alerts
- メッセージ形式: { "topic": "order.updates", "payload": {...}, "timestamp": ... }

7) 管理操作
- POST /api/v1/system/kill-switch
  - req: { "scope": "all|strategy:{id}|symbol:{sym}" }
  - resp: { "status": "ok" }

セキュリティ:
- TLS必須
- JWT/OAuth2アクセストークンによる認可
- 操作ログ（誰がいつ何をしたか）を必須出力

# 9. ログ設計
ログ分類:
- Audit Logs: 注文発行、取消、ユーザー操作、戦略の開始/停止（改竄防止のため改ざん検知/署名を検討）
- Trade Logs: 全Order/Fillイベント（order_id, client_order_id, exchange_order_id, symbol, side, qty, price, fees, timestamps）
- Market Data Logs: 原本受信ログ（raw）、正規化ログは軽量で保管
- System Logs: 各サービスのアプリケーションログ（level: DEBUG/INFO/WARN/ERROR）
- Metrics/Tracing: Prometheus metrics, OpenTelemetry traces

ログフォーマット:
- JSON 形式推奨（structured logging）
  - 共通フィールド: timestamp(ISO8601), service, instance_id, correlation_id, level, message, payload
  - correlation_id: トレードの一連の処理を追跡するため必ず付与

保存と保持:
- 高頻度データ（ティック）はTSDBまたは圧縮形式で短期保持（例: 90日）
- 取引/監査ログは長期保持（例: 7年、規制要件に従う）
- ログのアーカイブはS3へ転送。取り出しは監査用。

検索・分析:
- ELK/EFK でインデックス化し、検索・ダッシュボード・アラートに利用
- ClickHouse 等で大規模イベント分析（バックテスト結果等）

運用:
- ログの肥大化を防ぐため、INFO以下はレート制限、DEBUGはオンデマンドで有効化
- 異常検知ルール（例: エラー率が閾値超過 → PagerDuty通知）

# 10. リスク管理設計
リスクカテゴリ:
- 市場リスク: ポジションサイズ、最大許容損失、銘柄集中リスク
- 信用/ブローカーリスク: 取引所の残高・資金制約、接続断リスク
- オペレーショナルリスク: バグ、再現不能状態、データ欠損
- 技術リスク: レイテンシ、メッセージロス、単一障害点（SPOF）
- コンプライアンスリスク: 規制、監査、ログ保持

プリトレード（事前）チェック:
- 口座残高確認（available margin）
- 戦略別/グローバル注文量上限
- 1注文最大数量、1分あたり最大注文数
- 証拠金/レバレッジ制限
- 銘柄別トレード許可フラグ

ポストトレード（事後）監視:
- 取引後ポジション/想定P&Lと実績P&Lの乖離検知
- 異常約定（スリッページ超過、部分約定の偏り）
- 取引所からの拒否/遅延の頻発アラート

グローバル制御:
- Circuit Breaker:
  - システム全体/戦略単位/銘柄単位でトリガー（例: 一時間での損失が閾値超過 → 該当戦略を自動停止）
- Kill Switch:
  - 管理者/自動化監視から即時全取引停止
- フェイルオーバー:
  - Execution Gateway の障害時には待機ゲートウェイに切替、または全注文の自動キャンセル

フェイルセーフ:
- 注文再送ポリシー（冪等性のため client_order_id を利用）
- 未約定注文の自動再評価/キャンセル
- 監査可能な履歴（全注文・イベントの永続化）

テスト・検証:
- 戦略は必ずバックテスト・ヒストリカルシミュレーションで承認
- ストレステスト（マーケットデータ増大、遅延発生、API障害）
- カナリアデプロイ、段階的ロールアウト（新戦略は限定口座で稼働→監視→本番展開）

アラートと通知:
- 重大アラート（取引停止、回復不能なエラー）はオンコールへ（PagerDuty/Slack）
- 監視メトリクス: エラーレート、レイテンシ、スループット、未処理メッセージ、失敗注文率、P&L閾値

コンプライアンス & 監査:
- 全トレードは署名・タイムスタンプ付きで保持
- ロール毎に操作権限を制限（例: 取引実行は専用ロールのみ）
- 定期的なルール・ポリシー見直し（内部監査）

---

補足・次のステップ:
- 与えられた要件（{{$json.content}}）を反映して、以下を確定してください:
  - 対応市場（例: 仮想通貨/株式/FX）
  - レイテンシ要件（ターゲット数値）
  - データ保持ポリシー（期間）
  - 使用言語の優先（Python中心か低レイテンシ言語採用か）
  - 規模・同時戦略数の想定
- これらを頂ければ、さらに詳細なシーケンス図、ER図、APIスキーマ、Kubernetesマニフェスト例、CI/CDパイプライン設計を作成します。

必要に応じて、上記の各セクション（たとえば「外部API設計」をフルスキーマ化、または「リスク管理の閾値例」）を詳細化します。どの部分をより詳しく作成しますか？