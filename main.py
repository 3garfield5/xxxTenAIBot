from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os

load_dotenv('.env')

MODEL_PATH = os.environ.get('MODEL_PATH')
token=os.environ.get('BOT_TOKEN')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

async def start(update: Update, context):
    await update.message.reply_text("Hi bro! let's talk, say anything")


async def generate_text(update: Update, context):
    prompt = update.message.text

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=30,
            temperature=1,
            do_sample=True
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    await update.message.reply_text(generated_text)


def main():
    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_text))

    application.run_polling()


if __name__ == "__main__":
    main()