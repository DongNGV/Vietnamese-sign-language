import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="Tôi sẽ đưa cho bạn một chuỗi các từ. Nếu chuỗi có ít hơn 4 từ, hãy trả về chuỗi đó như cũ. Nếu chuỗi có từ 4 từ trở lên, hãy cố gắng chọn lọc hoặc thêm vài từ để tạo thành một câu giao tiếp hoàn chỉnh từ những từ đó. Nếu có thể tạo thành câu, hãy trả về câu đó. Nếu không thể tạo thành câu, hãy trả về chuỗi từ ban đầu. Luôn luôn chỉ đưa ra một câu trả lời duy nhất.",
)

# Tạo một chat session toàn cục và lưu giữ nó
_chat_session = model.start_chat(history=[])
# Tạo câu
def TaoCauHoanChinh(user_input):
    global _chat_session

    response = _chat_session.send_message(user_input)
    model_response = response.text
    return model_response