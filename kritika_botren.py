import logging
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram import Update
import os
import google.generativeai as genai

# --- Logger Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('TelegramBot')

# --- API Key Setup ---
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables.")

    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Successfully loaded API keys from environment variables.")

except ValueError as e:
    logger.critical(f"Failed to load API keys: {e}. Exiting.")
    import sys
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during API key setup: {e}", exc_info=True)
    import sys
    sys.exit(1)

# --- Gemini Model Configuration ---
GENERATION_CONFIG = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2500,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_instruction = (
    "You are Kritika, a friendly and supportive AI English teacher. Your role is to help Hindi-speaking users learn English step by step in 90 days. "
    "You explain English grammar in simple Hinglish (Hindi in English letters) and give practice translations daily. "
    "You speak politely, like a caring teacher, and focus on motivating the learner. "
    "Avoid complicated words unless you're teaching them. Be friendly, clear, and structured."
    "\n\n"
    "*Rules:* "
    "1. Speak in Hinglish or English based on the user's message. "
    "2. Give grammar explanations with examples. "
    "3. Include 10 sample translation sentences with sentence structure and English answers. "
    "4. Give 30 Hindi sentences for daily practice (without answers). "
    "5. Encourage the user with short motivational messages. "
    "6. Do not use adult, harmful, romantic, or abusive language. "
    "7. Do not pretend to be human—stay as Kritika the AI Teacher. "
    "\n\n"
    "*Emojis:* "
    "Use thoughtful emojis like 😊 (encouragement), 🤔 (thinking), 💡 (tips), 👍 (support) only when relevant. Avoid overuse."
)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS,
    system_instruction=system_instruction
)

conversations = {}

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_first_name = update.effective_user.first_name if update.effective_user else "दोस्त"
    conversations[chat_id] = model.start_chat(history=[])
    logger.info(f"Started new chat session for chat_id={chat_id}")
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"Hi {user_first_name}! 👋\n"
            "Main Kritika hoon – aapki English Teacher. 💡\n"
            "Main aapko 90 dino mein basic se advanced English sikhane wali hoon, step-by-step.\n"
            "Har din aapko grammar aur translation ka ek chhota task milega.\n"
            "Shuruaat karein? ✨"
        )
    )

async def generate_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_message = update.message.text if update.message else None
    user_first_name = update.effective_user.first_name if update.effective_user else "दोस्त"

    if not user_message:
        logger.info(f"Non-text or empty message from chat_id={chat_id}")
        return

    logger.info(f"Received message from chat_id={chat_id}: '{user_message}'")

    if chat_id not in conversations:
        conversations[chat_id] = model.start_chat(history=[])
        logger.info(f"Initialized new chat session for chat_id={chat_id}")

    chat_session = conversations[chat_id]

    try:
        # Add typing indicator for better UX while Gemini processes the request
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        prompt = f"उपयोगकर्ता का नाम: {user_first_name}\nउपयोगकर्ता का संदेश: {user_message}"
        logger.info(f"Sending to Gemini: '{prompt}'")

        # chat_session.send_message is synchronous in the default google-generativeai client
        response = chat_session.send_message(prompt)
        logger.info(f"Raw Gemini response: {str(response)}")

        bot_reply = ""
        if hasattr(response, "text") and response.text:
            bot_reply = response.text
        elif response.parts:
            for part in response.parts:
                if hasattr(part, 'text') and part.text:
                    bot_reply = part.text
                    break

        if bot_reply:
            logger.info(f"Sending reply: '{bot_reply}'")
            await context.bot.send_message(chat_id=chat_id, text=bot_reply)
        else:
            logger.warning(f"No text in Gemini response: {str(response)}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="माफ़ करें, कुछ गड़बड़ हो गई। मैं अभी जवाब नहीं दे पा रही हूँ।"
            )
    except Exception as e:
        logger.error(f"Error with Gemini API for chat_id={chat_id}: {e}", exc_info=True)
        await context.bot.send_message(
            chat_id=chat_id,
            text="क्षमा करें, मुझे जवाब देने में परेशानी हो रही है। कृपया बाद में कोशिश करें।"
        )

# --- Bot Setup ---
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
application.add_handler(CommandHandler('start', start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_response))

# --- Run the Bot ---
def main() -> None:
    """Start the bot and run it indefinitely until manually stopped."""
    logger.info("Starting bot with continuous polling...")
    print("Bot is now running.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutdown initiated.")
        print("\nBot has been stopped.")
    except Exception as e:
        logger.critical(f"Bot failed to run: {e}", exc_info=True)
        print(f"A critical error occurred: {e}")
