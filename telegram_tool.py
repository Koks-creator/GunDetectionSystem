from dataclasses import dataclass
import requests

from config import Config
from custom_logger import CustomLogger

logger = CustomLogger(logger_name="TelegramTool",
                      logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL
                      ).create_logger()


@dataclass
class TelegramTool:
    bot_token: str
    _api_url: str = "https://api.telegram.org/bot"

    def send_tg_message(self, msg: str, chat_id: str) -> None:
        url = f"{self._api_url}{self.bot_token}/sendMessage?" \
              f"chat_id={chat_id}&parse_mode=Markdown&text={msg}"
        response = requests.get(url)
        if response.status_code != 200:
            logger.error(f"Error when making sending tg message: {response.status_code}, {response.content}", exc_info=True)
    
    def send_tg_photo(self, photo_path: str, chat_id: str, caption: str = None) -> None:
        url = f"{self._api_url}{self.bot_token}/sendPhoto"
        
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, files=files, data=data)

            if response.status_code != 200:
                logger.error(f"Error when sending photo: {response.status_code}, {response.content}", exc_info=True)


if __name__ == "__main__":
    tg = TelegramTool(
        bot_token=Config.TELEGRAM_API_KEY
    )
    tg.send_tg_message(msg="Gunners: - per1", chat_id=Config.TELEGRAM_CHAT_ID)