import os
import io
import csv
import base64
import json
import gradio as gr
import pandas as pd
import concurrent.futures
import time
from PIL import Image
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal

# Load environment variables
load_dotenv()

# Initialize OpenAI client
environment = os.getenv("ENVIRONMENT")
if environment == "azure":
    client = AzureOpenAI(api_key=os.getenv("AZURE_API_KEY"), api_version="2024-10-21", azure_endpoint=os.getenv("AZURE_ENDPOINT"))
    MODEL = os.getenv("AZURE_MODEL")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = os.getenv("OPENAI_MODEL")

# Pydantic models for structured outputs
class ImageAnalysisResult(BaseModel):
    """Image analysis result with title and description"""
    title: str = Field(..., description="A concise, descriptive title for the image")
    description: str = Field(..., description="Detailed description of the image content, style, and possible use cases")

# 請求書分析用のPydanticモデル
class InvoiceItem(BaseModel):
    """請求書の項目"""
    name: str = Field(..., description="項目名/商品名")
    quantity: str = Field(..., description="数量")
    unit_price: str = Field(..., description="単価")
    amount: str = Field(..., description="金額")

class InvoiceAnalysisResult(BaseModel):
    """請求書から抽出した情報"""
    invoice_number: str = Field(..., description="請求書番号")
    issue_date: str = Field(..., description="発行日")
    due_date: str = Field(..., description="支払期限")
    company_name: str = Field(..., description="請求元会社名")
    total_amount: str = Field(..., description="合計金額")
    tax_amount: str = Field(..., description="税額")
    items: List[InvoiceItem] = Field(..., description="請求項目のリスト")
    notes: str = Field(..., description="その他特記事項")

# ソリューション定義
class Solution(BaseModel):
    """会社のソリューション定義"""
    id: str
    name: str
    description: str
    use_cases: List[str]
    
# サンプルソリューション定義
COMPANY_SOLUTIONS = [
    Solution(
        id="automation",
        name="製造ライン自動化",
        description="製造ラインの自動化ソリューションで、人手不足・コスト削減・品質向上を同時に実現します。",
        use_cases=["製造現場の効率化", "人手不足対策", "品質管理の向上"]
    ),
    Solution(
        id="security",
        name="セキュリティ対策",
        description="サイバー攻撃から企業を守る総合セキュリティ対策。オンプレミス環境からクラウド環境まで対応。",
        use_cases=["ランサムウェア対策", "クラウド移行時のセキュリティ", "リモートワーク環境のセキュリティ"]
    ),
    Solution(
        id="data_analytics",
        name="データ分析基盤",
        description="大量のデータから価値ある洞察を導き出すデータ分析基盤。顧客理解や市場予測に活用できます。",
        use_cases=["顧客データ分析", "マーケティング戦略立案", "セグメンテーション"]
    ),
    Solution(
        id="inventory",
        name="在庫管理システム",
        description="AIによる需要予測と連携した高度な在庫管理システム。複数店舗にも対応。",
        use_cases=["在庫の最適化", "需要予測", "複数店舗の在庫連携"]
    ),
    Solution(
        id="ai_poc",
        name="AI導入PoC",
        description="低コスト・短期間で実施可能なAI導入の実証実験（PoC）プログラム。",
        use_cases=["AI活用の検討", "デジタル化の第一歩", "小規模実証実験"]
    )
]

# ソリューション評価を表す列挙型
SolutionRating = Literal["", "✕", "△", "◯"]

# ソリューション評価モデル
class SolutionEvaluation(BaseModel):
    """各ソリューションの評価"""
    solution_id: str
    rating: SolutionRating
    reason: str

class SolutionSuggestion(BaseModel):
    """Solution suggestion for a sales visit entry"""
    reasoning: str
    evaluations: List[SolutionEvaluation] = Field(..., description="List of evaluations for each company solution")
    summary: str = Field(..., description="Brief summary of the analysis (around 150 characters)")

class SalesVisitEntry(BaseModel):
    """Raw data from a sales visit entry"""
    date: str
    company_name: str
    contact_person: str
    visit_notes: str

class SalesAnalysisResult(BaseModel):
    """Complete analysis of a sales visit entry"""
    row_data: SalesVisitEntry
    analysis: SolutionSuggestion

# Customer Support Models
class InquiryCategory(BaseModel):
    """問い合わせカテゴリ"""
    category: str = Field(..., description="問い合わせのカテゴリ (技術的問題, 返品/交換, 請求関連, 製品情報, 一般的質問 など)")
    confidence: float = Field(..., description="カテゴリ分類の確信度 (0.0 ~ 1.0)")

class EmotionAnalysis(BaseModel):
    """感情分析結果"""
    primary_emotion: str = Field(..., description="主要な感情 (怒り, 不満, 混乱, 心配, 中立, 期待, 満足 など)")
    urgency_level: str = Field(..., description="緊急度 (低, 中, 高)")
    tone: str = Field(..., description="トーン (フォーマル, カジュアル, ビジネスライク, 感情的 など)")

class KeyPoint(BaseModel):
    """キーポイント"""
    point: str = Field(..., description="問い合わせ内の重要ポイント")
    requires_information: bool = Field(..., description="追加情報が必要かどうか")

class InquiryAnalysis(BaseModel):
    """問い合わせ分析結果"""
    categories: List[InquiryCategory] = Field(..., description="問い合わせのカテゴリ (複数可)")
    emotion: EmotionAnalysis = Field(..., description="感情分析")
    key_points: List[KeyPoint] = Field(..., description="キーポイント")
    summary: str = Field(..., description="問い合わせの要約 (100文字以内)")

class ResponseDraft(BaseModel):
    """回答案"""
    formal_response: str = Field(..., description="フォーマルな回答文")
    required_followup: Optional[str] = Field(None, description="必要なフォローアップアクション")
    references: List[str] = Field(..., description="参照すべき資料やナレッジベース")

class CustomerInquiry(BaseModel):
    """顧客問い合わせ情報"""
    inquiry_id: str
    customer_name: str
    customer_type: str
    product_category: str
    inquiry_text: str

class ProcessedInquiry(BaseModel):
    """処理済み問い合わせ"""
    inquiry: CustomerInquiry
    analysis: InquiryAnalysis
    adjusted_categories: Optional[List[str]] = None
    adjusted_urgency: Optional[str] = None
    additional_context: Optional[str] = None
    response: Optional[ResponseDraft] = None

def analyze_image(image):
    """
    請求書画像から情報を抽出する
    """
    if image is None:
        return None, "画像がアップロードされていません"
    
    try:
        # 画像をbase64に変換
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # OpenAI APIでPydanticモデルを使用して構造化データを取得
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは請求書から情報を抽出する専門家です。アップロードされた請求書画像から、請求書番号、日付、金額、項目などの重要情報を正確に抽出してください。"
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "この請求書画像から情報を抽出してください。請求書番号、発行日、支払期限、請求元、金額、税額、請求項目などを識別してください。情報が見つからない場合は「不明」と記入してください。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                    ]
                }
            ],
            response_format=InvoiceAnalysisResult,
            max_tokens=1000
        )
        
        # 解析結果を取得
        result = response.choices[0].message.parsed
        
        # 結果を整形してHTML形式で返す
        html_output = f"""
        <div style='background-color: #f9f9f9; padding: 20px; border-radius: 8px; font-family: Arial, sans-serif;'>
            <h3 style='color: #333; margin-top: 0;'>請求書情報</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='font-weight: bold; padding: 8px; border-bottom: 1px solid #ddd;'>請求書番号</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{result.invoice_number}</td>
                </tr>
                <tr>
                    <td style='font-weight: bold; padding: 8px; border-bottom: 1px solid #ddd;'>発行日</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{result.issue_date}</td>
                </tr>
                <tr>
                    <td style='font-weight: bold; padding: 8px; border-bottom: 1px solid #ddd;'>支払期限</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{result.due_date}</td>
                </tr>
                <tr>
                    <td style='font-weight: bold; padding: 8px; border-bottom: 1px solid #ddd;'>請求元</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{result.company_name}</td>
                </tr>
                <tr>
                    <td style='font-weight: bold; padding: 8px; border-bottom: 1px solid #ddd;'>合計金額</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{result.total_amount}</td>
                </tr>
                <tr>
                    <td style='font-weight: bold; padding: 8px; border-bottom: 1px solid #ddd;'>税額</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{result.tax_amount}</td>
                </tr>
            </table>
            
            <h4 style='color: #333; margin-top: 20px;'>請求項目</h4>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='background-color: #eee;'>
                    <th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>項目</th>
                    <th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>数量</th>
                    <th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>単価</th>
                    <th style='padding: 8px; text-align: left; border-bottom: 1px solid #ddd;'>金額</th>
                </tr>
        """
        
        # 請求項目を追加
        for item in result.items:
            html_output += f"""
                <tr>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{item.name}</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{item.quantity}</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{item.unit_price}</td>
                    <td style='padding: 8px; border-bottom: 1px solid #ddd;'>{item.amount}</td>
                </tr>
            """
        
        html_output += """
            </table>
        """
        
        # 備考があれば追加
        if result.notes:
            # 改行を<br>に変換
            notes_html = result.notes.replace('\n', '<br>')
            html_output += f"""
            <h4 style='color: #333; margin-top: 20px;'>備考</h4>
            <p style='margin-top: 5px;'>{notes_html}</p>
            """
        
        html_output += "</div>"
        
        return html_output, ""
    
    except Exception as e:
        return None, f"請求書の分析中にエラーが発生しました: {str(e)}"

def analyze_sales_report(csv_file, progress=gr.Progress()):
    """
    Analyze the sales visit report CSV file and suggest potential solutions using Structured Outputs
    With parallel processing and progress tracking
    """
    if csv_file is None:
        return "No file uploaded", None
    
    try:
        # Read the CSV file
        if isinstance(csv_file, str):  # If it's a file path
            df = pd.read_csv(csv_file)
        else:  # If it's a file object from Gradio
            csv_content = csv_file.decode('utf-8') if isinstance(csv_file, bytes) else csv_file
            df = pd.read_csv(io.StringIO(csv_content))
        
        # Calculate total number of rows to process
        total_rows = len(df)
        progress(0, desc="準備中...")
        
        # Function to analyze a single row
        def analyze_row(row_tuple):
            idx, row = row_tuple
            try:
                # Convert row to Pydantic model
                row_data = SalesVisitEntry(
                    date=str(row['date']),
                    company_name=str(row['company_name']),
                    contact_person=str(row['contact_person']),
                    visit_notes=str(row['visit_notes'])
                )
                
                # Use OpenAI to analyze the row with Structured Outputs
                visit_text = f"日付: {row_data.date}, 会社名: {row_data.company_name}, 担当者: {row_data.contact_person}, 訪問メモ: {row_data.visit_notes}"
                
                # 利用可能なソリューション情報を作成
                solutions_info = "\n".join([
                    f"ID: {sol.id}, 名前: {sol.name}, 説明: {sol.description}, ユースケース: {', '.join(sol.use_cases)}"
                    for sol in COMPANY_SOLUTIONS
                ])
                
                response = client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": f"""
                        あなたは営業支援アドバイザーです。訪問メモから最適なソリューションを提案してください。
                        
                        以下のソリューションがあります：
                        {solutions_info}
                        
                        各ソリューションについて、訪問メモに基づいて4段階で評価してください:
                        - 空白 ("") = 関心がない、訪問メモから判断できない（情報不足）
                        - "✕" = 非推奨（顧客が明確にソリューションや分野に強く否定的、拒否の姿勢）
                        - "△" = 可能性あり（ある程度関心を持ちそうなソリューション）
                        - "◯" = 可能性大（非常に関心を持ちそうなソリューション）
                        
                        各評価には簡潔な理由も付けてください。
                        また、全体を150文字程度で要約してください。
                        """},
                        {"role": "user", "content": f"以下の営業訪問記録を分析し、適切なソリューションを提案してください: {visit_text}"}
                    ],
                    response_format=SolutionSuggestion,
                    max_tokens=1000
                )
                
                # Get the parsed Pydantic model directly
                analysis = response.choices[0].message.parsed
                
                # Create complete analysis result
                result = SalesAnalysisResult(
                    row_data=row_data,
                    analysis=analysis
                )
                
                return idx, result, None  # Success
            except Exception as e:
                return idx, None, f"行 {idx+1} の処理中にエラー: {str(e)}"  # Error
        
        # Prepare for parallel processing
        all_results = [None] * total_rows
        errors = []
        
        # Use ThreadPoolExecutor for parallel API calls
        # Set max_workers to 5 or less to avoid rate limiting
        max_workers = min(5, total_rows)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(analyze_row, (idx, row)): idx for idx, row in df.iterrows()}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                idx, result, error = future.result()
                progress((i + 1) / total_rows, desc=f"分析中... {i + 1}/{total_rows}")
                
                if error:
                    errors.append(error)
                else:
                    all_results[idx] = result
            
        # Filter out None values (failed analyses)
        results = [r for r in all_results if r is not None]
        
        # Create a table HTML output
        html_output = """
        <div style='max-height: 500px; overflow-y: auto; margin-top: 20px;'>
        """
        
        # Show errors if any
        if errors:
            html_output += """
            <div style='margin-bottom: 20px; padding: 10px; background-color: #fff0f0; border-left: 4px solid #f44336; border-radius: 4px;'>
                <h3 style='margin-top: 0; color: #d32f2f;'>処理中に発生したエラー</h3>
                <ul>
            """
            for error in errors:
                html_output += f"<li>{error}</li>"
            html_output += "</ul></div>"
        
        # Create the results table
        html_output += """
        <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
        <thead>
            <tr style='background-color: #f2f2f2;'>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>No</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>日付</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>会社名</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>担当者</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>訪問メモ</th>
        """
        
        # Add solution headers
        for solution in COMPANY_SOLUTIONS:
            html_output += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{solution.name}</th>"
        
        # Add analysis header
        html_output += "<th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>分析結果</th></tr></thead><tbody>"
        
        # Add rows
        for i, result in enumerate(results):
            html_output += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{i+1}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{result.row_data.date}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{result.row_data.company_name}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{result.row_data.contact_person}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{result.row_data.visit_notes[:100]}...</td>
            """
            
            # 各ソリューションの評価を追加
            solution_rating_map = {eval.solution_id: eval for eval in result.analysis.evaluations}
            
            for solution in COMPANY_SOLUTIONS:
                evaluation = solution_rating_map.get(solution.id)
                if evaluation:
                    rating_str = evaluation.rating
                    tooltip = evaluation.reason
                    html_output += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center; font-size: 18px;' title='{tooltip}'>{rating_str}</td>"
                else:
                    html_output += "<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'></td>"
            
            # 分析結果を追加
            html_output += f"<td style='padding: 8px; border: 1px solid #ddd;'>{result.analysis.summary}</td></tr>"
        
        html_output += "</tbody></table></div>"
        
        # ソリューション説明テーブルを追加
        html_output += """
        <div style='margin-top: 30px;'>
        <h3>ソリューション説明</h3>
        <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
        <thead>
            <tr style='background-color: #f2f2f2;'>
                <th style='padding: 8px; border: 1px solid #ddd;'>ソリューション名</th>
                <th style='padding: 8px; border: 1px solid #ddd;'>説明</th>
                <th style='padding: 8px; border: 1px solid #ddd;'>主な用途</th>
            </tr>
        </thead>
        <tbody>
        """
        
        for solution in COMPANY_SOLUTIONS:
            html_output += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #ddd;'>{solution.name}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{solution.description}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{", ".join(solution.use_cases)}</td>
            </tr>
            """
        
        html_output += "</tbody></table></div>"
        
        # 評価基準の説明を追加
        html_output += """
        <div style='margin-top: 20px;'>
        <h3>評価基準</h3>
        <ul>
            <li>◯: 可能性大 - 非常に関心を持ちそうなソリューション</li>
            <li>△: 可能性あり - ある程度関心を持ちそうなソリューション</li>
            <li>✕: 非推奨 - 顧客が明確にそのソリューションや分野に否定的である</li>
            <li>(空白): 情報不足 - 訪問メモから判断するための十分な情報がない</li>
        </ul>
        </div>
        """
        
        status_message = f"分析完了: {len(results)}/{total_rows} 件の処理に成功"
        if errors:
            status_message += f" ({len(errors)} 件のエラー)"
        
        return status_message, html_output
    
    except Exception as e:
        return f"Error processing file: {str(e)}", None

def analyze_customer_inquiry(inquiry, progress=None, index=0, total=1):
    """
    問い合わせを分析し、カテゴリ、感情、キーポイントを抽出します
    """
    try:
        # 進捗表示の処理を安全に行う
        if progress is not None:
            try:
                progress((index / total), desc=f"分析中 {index+1}/{total}...")
            except (IndexError, AttributeError, Exception) as e:
                # プログレスバーのエラーは無視して処理を続行
                print(f"Progress update error (non-critical): {str(e)}")
        
        # 前段LLM処理: 問い合わせ分析
        prompt = f"""
        以下の顧客問い合わせを分析してください:
        
        顧客名: {inquiry.customer_name}
        顧客タイプ: {inquiry.customer_type}
        製品カテゴリ: {inquiry.product_category}
        問い合わせ内容: {inquiry.inquiry_text}
        """
        
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": """あなたはカスタマーサポートアナリストです。
                顧客からの問い合わせを分析し、そのカテゴリ、感情、重要ポイントを識別してください。
                分析は客観的で正確である必要があります。"""},
                {"role": "user", "content": prompt}
            ],
            response_format=InquiryAnalysis,
            max_tokens=1000
        )
        
        analysis = response.choices[0].message.parsed
        return ProcessedInquiry(
            inquiry=inquiry,
            analysis=analysis,
            adjusted_categories=None,
            adjusted_urgency=None,
            additional_context=None,
            response=None
        )
    
    except Exception as e:
        print(f"Error analyzing inquiry {inquiry.inquiry_id}: {str(e)}")
        return None

def generate_response(inquiry_id, category, urgency, additional_context, processed_data_json):
    """選択したカテゴリと緊急度に基づいて回答を生成"""
    print(f"回答生成開始: ID={inquiry_id}, カテゴリ={category}, 緊急度={urgency}")
    
    if not inquiry_id or not processed_data_json:
        return "問い合わせデータが不足しています。", None
    
    try:
        # JSON文字列からデータを復元
        inquiries = []
        for json_str in processed_data_json:
            inquiry_dict = json.loads(json_str)
            inquiry = ProcessedInquiry.model_validate(inquiry_dict)
            inquiries.append(inquiry)
        
        # 選択された問い合わせを検索
        selected = None
        for inq in inquiries:
            if inq.inquiry.inquiry_id == inquiry_id:
                selected = inq
                break
        
        if not selected:
            return "選択した問い合わせが見つかりません。", None
        
        # カテゴリと緊急度を更新（is_selectedフィールドは使わない）
        # 代わりに選択されたカテゴリを直接使用
        selected_categories = [cat.category for cat in selected.analysis.categories if cat.category == category]
        
        # 緊急度を更新
        selected.analysis.emotion.urgency_level = urgency
        
        # 追加コンテキストを更新
        if additional_context:
            selected.additional_context = additional_context
        
        # LLMを使用して回答を生成
        print("LLMを使用して回答を生成します...")
        user_message = f"""
問い合わせID: {selected.inquiry.inquiry_id}
顧客名: {selected.inquiry.customer_name}
顧客タイプ: {selected.inquiry.customer_type}
製品カテゴリ: {selected.inquiry.product_category}
問い合わせ内容: {selected.inquiry.inquiry_text}

選択されたカテゴリ: {category}
緊急度: {urgency}

キーポイント:
{chr(10).join([f"- {kp.point}" + (" (要追加情報)" if kp.requires_information else "") for kp in selected.analysis.key_points])}

感情分析:
- 主要な感情: {selected.analysis.emotion.primary_emotion}
- トーン: {selected.analysis.emotion.tone}

{f"追加コンテキスト: {selected.additional_context}" if selected.additional_context else ""}

上記の情報を元に、顧客へのサポート回答のメール文面を作成してください。
回答は敬語を使用し、共感的で問題解決に焦点を当て、可能な限り具体的な解決策や次のステップを含めてください。
また、フォローアップに必要な情報がある場合は、それを明記してください。
件名等を含まないメール文面のみを出力してください。
"""
        
        messages = [
            {"role": "system", "content": "あなたはプロフェッショナルなカスタマーサポート担当者です。丁寧で共感的、かつ問題解決志向の回答を提供します。"},
            {"role": "user", "content": user_message}
        ]
        
        # OpenAI APIを使用して回答を生成
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        
        # 回答を取得
        reply = response.choices[0].message.content
        
        # フォローアップ情報の抽出
        system_prompt = """
以下の回答を分析し、フォローアップに必要な情報や次のアクションを、<br>タグで改行された箇条書きで記載してください。
たとえば、「電話番号の確認が必要」「製品のシリアル番号を確認」「技術部門への転送を検討」などの重要なアクションポイントを抽出します。
3〜5点程度にまとめてください。
"""
        
        followup_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reply}
        ]
        
        followup_response = client.chat.completions.create(
            model=MODEL,
            messages=followup_messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        followup_text = followup_response.choices[0].message.content
        
        # 整形したフォローアップ情報
        formatted_followup = f"""
<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin-top: 20px;">
<h4 style="margin-top: 0;">フォローアップポイント:</h4>
{followup_text}
</div>
"""
        
        # 回答を返す
        return reply, formatted_followup
    
    except Exception as e:
        print(f"回答生成中のエラー: {e}")
        import traceback
        traceback.print_exc()
        return f"回答の生成中にエラーが発生しました: {str(e)}", None

def process_customer_inquiries(csv_file, progress=gr.Progress()):
    """
    CSVファイルから顧客問い合わせを読み込み、分析します
    """
    if csv_file is None:
        return "CSVファイルがアップロードされていません", None, None, []
    
    try:
        # CSVファイルを読み込み
        if isinstance(csv_file, str):
            df = pd.read_csv(csv_file)
        else:
            csv_content = csv_file.decode('utf-8') if isinstance(csv_file, bytes) else csv_file
            df = pd.read_csv(io.StringIO(csv_content))
        
        # データフレームの列名を確認し、必要に応じて変換
        required_columns = ['inquiry_id', 'customer_name', 'customer_type', 'product_category', 'inquiry_text']
        if not all(col in df.columns for col in required_columns):
            # 日本語の列名の場合のマッピング
            jp_columns = {
                '問い合わせID': 'inquiry_id',
                '顧客名': 'customer_name',
                '顧客タイプ': 'customer_type',
                '製品カテゴリ': 'product_category',
                '問い合わせ内容': 'inquiry_text'
            }
            
            # 列名変換を試みる
            for jp_col, en_col in jp_columns.items():
                if jp_col in df.columns and en_col not in df.columns:
                    df = df.rename(columns={jp_col: en_col})
        
        # 必要な列が全てあるか確認
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return f"CSVファイルに必要な列がありません: {', '.join(missing_cols)}", None, None, []
        
        # CustomerInquiry オブジェクトのリストに変換
        inquiries = []
        for _, row in df.iterrows():
            inquiry = CustomerInquiry(
                inquiry_id=str(row['inquiry_id']),
                customer_name=str(row['customer_name']),
                customer_type=str(row['customer_type']),
                product_category=str(row['product_category']),
                inquiry_text=str(row['inquiry_text'])
            )
            inquiries.append(inquiry)
        
        # 進捗状況を初期化
        progress(0, desc="分析準備中...")
        
        # 分析を並列実行
        total = len(inquiries)
        max_workers = min(5, total)  # 最大5件を並列処理
        processed_inquiries = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, inquiry in enumerate(inquiries):
                    future = executor.submit(analyze_customer_inquiry, inquiry, progress, i, total)
                    futures[future] = i
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            processed_inquiries.append(result)
                    except Exception as e:
                        print(f"Error processing inquiry: {str(e)}")
                        # エラーが発生しても処理を継続
        except Exception as e:
            print(f"Thread pool execution error: {str(e)}")
            # 少なくとも部分的な結果を返すために続行
        
        # inquiry_id でソート
        processed_inquiries.sort(key=lambda x: x.inquiry.inquiry_id)
        
        # HTML形式の一覧表を生成
        html_table = """
        <div style='max-height: 500px; overflow-y: auto; margin-top: 20px;'>
        <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
        <thead>
            <tr style='background-color: #f2f2f2;'>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>ID</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>顧客名</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>製品</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>問い合わせ概要</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>カテゴリ</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>緊急度</th>
                <th style='padding: 8px; border: 1px solid #ddd; text-align: center;'>感情</th>
            </tr>
        </thead>
        <tbody>
        """
        
        # 行を追加
        for i, inquiry in enumerate(processed_inquiries):
            # カテゴリのリスト
            categories = [f"{cat.category} ({int(cat.confidence*100)}%)" for cat in inquiry.analysis.categories]
            
            # 感情と緊急度
            emotion = inquiry.analysis.emotion.primary_emotion
            urgency = inquiry.analysis.emotion.urgency_level
            urgency_color = {
                "低": "#5cb85c",  # 緑
                "中": "#f0ad4e",  # オレンジ
                "高": "#d9534f"   # 赤
            }.get(urgency, "#5bc0de")  # デフォルト: 青
            
            # 問い合わせ概要
            summary = inquiry.analysis.summary
            
            html_table += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{inquiry.inquiry.inquiry_id}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{inquiry.inquiry.customer_name}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{inquiry.inquiry.product_category}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{summary}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{", ".join(categories)}</td>
                <td style='padding: 8px; border: 1px solid #ddd; text-align: center; background-color: {urgency_color}; color: white;'>{urgency}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{emotion}</td>
            </tr>
            """
        
        html_table += "</tbody></table></div>"
        
        # ドロップダウンの選択肢を作成
        inquiry_ids = []
        for inquiry in processed_inquiries:
            inquiry_ids.append(inquiry.inquiry.inquiry_id)
        
        # ステータスメッセージ
        status = f"{len(processed_inquiries)}件の問い合わせを分析しました"
        
        # 処理済み問い合わせとして保持するためにJSONに変換
        # より安全なJSON処理のために辞書に変換してからjsondumpsを使用
        processed_data = []
        for inquiry in processed_inquiries:
            try:
                # 新しいPydanticメソッドを使用
                inquiry_dict = json.loads(inquiry.model_dump_json())
                processed_data.append(json.dumps(inquiry_dict))
            except Exception as e:
                print(f"JSON conversion error for {inquiry.inquiry.inquiry_id}: {str(e)}")
        
        return status, html_table, inquiry_ids, processed_data
    
    except Exception as e:
        return f"エラー: {str(e)}", None, None, []

def show_inquiry_details(inquiry_id, processed_data_json):
    """選択された問い合わせの詳細を表示"""
    print(f"問い合わせ詳細表示: ID={inquiry_id}")
    
    if not inquiry_id or not processed_data_json:
        print("問い合わせIDまたは処理済みデータがありません")
        return None, None, None, None, None, None
    
    try:
        # JSON文字列からデータを復元
        inquiries = []
        for json_str in processed_data_json:
            inquiry_dict = json.loads(json_str)
            inquiry = ProcessedInquiry.model_validate(inquiry_dict)
            inquiries.append(inquiry)
        
        # 選択された問い合わせを検索
        print(f"検索対象ID: {inquiry_id}")
        print(f"利用可能なID: {[inq.inquiry.inquiry_id for inq in inquiries]}")
        
        selected = None
        for inq in inquiries:
            if inq.inquiry.inquiry_id == inquiry_id:
                selected = inq
                print(f"問い合わせが見つかりました: {inquiry_id}")
                break
        
        if not selected:
            print(f"問い合わせが見つかりません: {inquiry_id}")
            return None, None, None, None, None, None
        
        # カテゴリ
        categories = [cat.category for cat in selected.analysis.categories]
        all_categories = [
            "技術的問題", "返品/交換", "請求関連", "製品情報", "一般的質問", 
            "アカウント管理", "配送状況", "使い方", "クレーム", "その他"
        ]
        category_choices = [cat for cat in all_categories if cat in categories]
        
        # 緊急度
        urgency = selected.analysis.emotion.urgency_level
        
        # キーポイント
        key_points = "\n".join([f"- {kp.point}" + (" (要追加情報)" if kp.requires_information else "") 
                           for kp in selected.analysis.key_points])
        
        # 感情分析
        emotion_html = f"""
        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <h4 style='margin-top: 0;'>感情分析結果</h4>
            <ul>
                <li><strong>主要な感情:</strong> {selected.analysis.emotion.primary_emotion}</li>
                <li><strong>トーン:</strong> {selected.analysis.emotion.tone}</li>
                <li><strong>緊急度:</strong> {selected.analysis.emotion.urgency_level}</li>
            </ul>
        </div>
        """
        
        # 追加コンテキスト
        additional_context = selected.additional_context or ""
        
        return (
            selected.inquiry.inquiry_text,
            category_choices,
            urgency,
            key_points,
            emotion_html,
            additional_context
        )
    
    except Exception as e:
        print(f"問い合わせ詳細表示中のエラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def adjust_and_generate(inquiry_id, adjusted_categories, adjusted_urgency, additional_context, processed_data_json, progress=gr.Progress()):
    """選択した問い合わせの分析と設定を更新し、回答を生成します"""
    print(f"adjust_and_generate called with inquiry_id: {inquiry_id}")
    print(f"Categories: {adjusted_categories}, Urgency: {adjusted_urgency}")
    
    if not inquiry_id or not processed_data_json:
        return "データがありません", "", "", "", processed_data_json
    
    try:
        # 処理中の表示
        progress(0.1, desc="問い合わせ情報を更新しています...")
        
        # カテゴリがリストの場合は最初の要素を取得
        category = adjusted_categories[0] if isinstance(adjusted_categories, list) and adjusted_categories else adjusted_categories
        
        # 回答を生成
        progress(0.4, desc="回答を生成しています...")
        formal_response, followup_info = generate_response(inquiry_id, category, adjusted_urgency, additional_context, processed_data_json)
                
        # 進捗完了
        progress(1.0, desc="完了")
        
        return "回答を生成しました", formal_response, followup_info, processed_data_json
    except Exception as e:
        print(f"回答生成中のエラー: {e}")
        import traceback
        traceback.print_exc()
        # エラー発生時も進捗バーを完了させる
        progress(1.0, desc="エラーが発生しました")
        return f"エラー: {str(e)}", "", "", "", processed_data_json

# Create Gradio interface
with gr.Blocks(title="LLMビジネスアプリケーション デモ") as demo:
    gr.Markdown("# LLMビジネスアプリケーション デモ")
    gr.Markdown("ビジネス向け大規模言語モデルの非チャットアプリケーションを探索する")
    
    with gr.Tab("請求書解析"):
        gr.Markdown("## 請求書画像解析")
        gr.Markdown("請求書画像をアップロードすると、AI技術を使って重要情報（請求書番号、発行日、金額、項目など）を自動的に抽出します。請求書処理の効率化、データ入力の自動化に活用できます。")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="請求書画像をアップロード")
                analyze_button = gr.Button("請求書を解析", variant="primary")
            
            with gr.Column(scale=2):
                result_output = gr.HTML(label="抽出結果")
                error_output = gr.Textbox(label="エラーメッセージ", visible=True)
        
        analyze_button.click(
            analyze_image,
            inputs=[image_input],
            outputs=[result_output, error_output]
        )

    with gr.Tab("営業訪問分析"):
        gr.Markdown("## 営業訪問記録分析")
        gr.Markdown("""
        営業訪問メモから最適なソリューションを自動的に分析します。
        CSVファイルをアップロードすると、各訪問に対して当社ソリューションの適合度を4段階で評価します。
        並列処理により複数の訪問データを同時に分析し、短時間で結果を得ることができます。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                csv_input = gr.File(label="営業訪問CSVをアップロード", file_types=[".csv"])
                with gr.Row():
                    analyze_csv_button = gr.Button("分析開始", variant="primary", scale=3)
                    cancel_button = gr.Button("キャンセル", variant="stop", scale=1)
                
                status_output = gr.Textbox(label="状態", value="ファイルをアップロードして「分析開始」をクリックしてください")
            
            with gr.Column(scale=2):
                results_output = gr.HTML(label="分析結果")
        
        # リセット機能
        def reset_analysis():
            return "ファイルをアップロードして「分析開始」をクリックしてください", None
        
        # 分析実行
        analyze_job = analyze_csv_button.click(
            analyze_sales_report,
            inputs=[csv_input],
            outputs=[status_output, results_output],
            show_progress=True,  # これにより進捗バーが表示されます
        )
        
        # キャンセルボタンの設定
        cancel_button.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[analyze_job]
        )
        
        # 新しいCSVファイルがアップロードされたらリセット
        csv_input.change(
            fn=reset_analysis,
            inputs=None,
            outputs=[status_output, results_output]
        )

    with gr.Tab("カスタマーサポート回答生成"):
        gr.Markdown("## カスタマーサポート回答生成システム")
        gr.Markdown("""
        顧客からの問い合わせを分析し、適切な回答案を生成するシステムです。
        CSVファイルから複数の問い合わせを一括読み込み、AIが事前分析を行います。
        オペレーターは分析結果を確認・調整した上で、最適な回答を生成できます。
        """)
        
        # 保存用の状態変数
        processed_data_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                # 入力セクション
                cs_csv_input = gr.File(label="問い合わせCSVファイル", file_types=[".csv"])
                cs_analyze_button = gr.Button("分析開始", variant="primary")
                cs_status_output = gr.Textbox(label="状態", value="ファイルをアップロードして「分析開始」をクリックしてください")
                
                # 問い合わせ選択
                gr.Markdown("### 問い合わせ詳細")
                inquiry_selector = gr.Dropdown(
                    choices=[], 
                    label="問い合わせを選択", 
                    interactive=True,
                    allow_custom_value=False
                )
                
                # 調整セクション
                category_selector = gr.CheckboxGroup(
                    choices=[
                        "技術的問題", "返品/交換", "請求関連", "製品情報", "一般的質問", 
                        "アカウント管理", "配送状況", "使い方", "クレーム", "その他"
                    ],
                    label="カテゴリ調整",
                    interactive=True
                )
                urgency_selector = gr.Radio(
                    choices=["低", "中", "高"],
                    label="緊急度調整",
                    interactive=True
                )
                additional_context = gr.Textbox(
                    label="追加コンテキスト",
                    placeholder="対応履歴や追加情報があればここに入力してください",
                    lines=3,
                    interactive=True
                )
                generate_button = gr.Button("回答生成", variant="primary")
            
            with gr.Column(scale=2):
                # 一覧表示セクション
                cs_results_output = gr.HTML(label="分析結果一覧")
                
                # 選択された問い合わせの詳細表示
                with gr.Row():
                    with gr.Column(scale=1):
                        inquiry_text = gr.Textbox(label="問い合わせ内容", lines=6, interactive=False)
                    with gr.Column(scale=1):
                        key_points = gr.Textbox(label="キーポイント", lines=6, interactive=False)
                
                emotion_output = gr.HTML(label="感情分析")
                
                # 生成された回答セクション
                with gr.Accordion("生成された回答", open=False) as response_accordion:
                    response_status = gr.Textbox(label="生成状態", value="", visible=True)
                    formal_response = gr.Textbox(label="回答", lines=8, interactive=True)
                    followup_info = gr.HTML(label="フォローアップ")
        
        # CSVアップロード・分析のイベント連携
        analyze_result = cs_analyze_button.click(
            process_customer_inquiries,
            inputs=[cs_csv_input],
            outputs=[cs_status_output, cs_results_output, inquiry_selector, processed_data_state],
            show_progress=True
        )
        
        # カスタム表示の初期化関数
        def init_first_inquiry(status, html, ids, processed_data):
            if ids and len(ids) > 0:
                first_id = ids[0]
                print(f"初期化関数: 最初のID={first_id}を選択")
                
                # 最初の問い合わせの詳細を取得
                try:
                    inquiries = []
                    for json_str in processed_data:
                        inquiry_dict = json.loads(json_str)
                        inquiry = ProcessedInquiry.model_validate(inquiry_dict)
                        inquiries.append(inquiry)
                    
                    # 最初の問い合わせを検索
                    selected = next((inq for inq in inquiries if inq.inquiry.inquiry_id == first_id), None)
                    
                    if selected:
                        # カテゴリ
                        categories = [cat.category for cat in selected.analysis.categories]
                        all_categories = [
                            "技術的問題", "返品/交換", "請求関連", "製品情報", "一般的質問", 
                            "アカウント管理", "配送状況", "使い方", "クレーム", "その他"
                        ]
                        category_choices = [cat for cat in all_categories if cat in categories]
                        
                        # 緊急度
                        urgency = selected.analysis.emotion.urgency_level
                        
                        # キーポイント
                        key_points = "\n".join([f"- {kp.point}" + (" (要追加情報)" if kp.requires_information else "") 
                                           for kp in selected.analysis.key_points])
                        
                        # 感情分析
                        emotion_html = f"""
                        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 10px;'>
                            <h4 style='margin-top: 0;'>感情分析結果</h4>
                            <ul>
                                <li><strong>主要な感情:</strong> {selected.analysis.emotion.primary_emotion}</li>
                                <li><strong>トーン:</strong> {selected.analysis.emotion.tone}</li>
                                <li><strong>緊急度:</strong> {selected.analysis.emotion.urgency_level}</li>
                            </ul>
                        </div>
                        """
                        
                        # 追加コンテキスト
                        additional_context = selected.additional_context or ""
                        
                        return (
                            gr.update(choices=ids, value=first_id),
                            selected.inquiry.inquiry_text,
                            category_choices,
                            urgency,
                            key_points,
                            emotion_html,
                            additional_context
                        )
                except Exception as e:
                    print(f"初期化中のエラー: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 問い合わせがない場合や処理中にエラーが発生した場合
            return gr.update(choices=ids), None, None, None, None, None, None
        
        # 分析完了後に最初の問い合わせを選択して詳細を表示
        analyze_result.then(
            init_first_inquiry,
            inputs=[cs_status_output, cs_results_output, inquiry_selector, processed_data_state],
            outputs=[inquiry_selector, inquiry_text, category_selector, urgency_selector, key_points, emotion_output, additional_context]
        )
        
        # 問い合わせ選択時の詳細表示
        inquiry_selector.change(
            show_inquiry_details,
            inputs=[inquiry_selector, processed_data_state],
            outputs=[inquiry_text, category_selector, urgency_selector, key_points, emotion_output, additional_context]
        )
        
        # 回答生成のイベント連携
        generate_button.click(
            adjust_and_generate,
            inputs=[inquiry_selector, category_selector, urgency_selector, additional_context, processed_data_state],
            outputs=[cs_status_output, formal_response, followup_info, processed_data_state],
            show_progress=True
        ).then(
            lambda: gr.update(open=True),
            inputs=None,
            outputs=[response_accordion]
        )
        
        # CSVのサンプル
        gr.Markdown("""
        ### サンプルCSVフォーマット
        以下の形式のCSVファイルをアップロードしてください。
        
        ```
        inquiry_id,customer_name,customer_type,product_category,inquiry_text
        CS001,山田太郎,既存顧客,ソフトウェアA,インストール後にエラーが発生します。エラーコードは「E-1234」です。早急に対応してください。
        CS002,鈴木花子,見込み客,ハードウェアB,価格と納期について教えてください。10台まとめて購入予定です。
        ```
        
        日本語の列名でも対応可能です（問い合わせID,顧客名,顧客タイプ,製品カテゴリ,問い合わせ内容）
        """)

if __name__ == "__main__":
    demo.launch(server_port=17860)
