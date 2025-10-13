import json
import requests
import re


class TelegramService:
    """
    Telegram service for sending messages to a specific chat.
    """

    def __init__(self, bot_token: str, chat_id: int = 1569280286):
        """
        Initialize the Telegram service.

        Args:
            bot_token (str): The Telegram bot token
            chat_id (int): The chat ID to send messages to (defaults to existing chat)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/{bot_token}/sendMessage"

        # Emoji constants
        self.warning = '⚠️'
        self.success = '✅'
        self.error = '❌'

    @staticmethod
    def escape_markdown(text: str, entity_type: str | None = None) -> str:
        """
        Helper function to escape telegram markup symbols.
        Args:
            text (:obj:`str`): The text.
            version (:obj:`int` | :obj:`str`): Use to specify the version of telegrams Markdown.
                Either ``1`` or ``2``. Defaults to ``1``.
            entity_type (:obj:`str`, optional): For the entity types ``PRE``, ``CODE`` and the link
                part of ``TEXT_LINKS``, only certain characters need to be escaped in ``MarkdownV2``.
                See the official API documentation for details. Only valid in combination with
                ``version=2``, will be ignored else.
        """
        if entity_type in ['pre', 'code']:
            escape_chars = r'\`'
        elif entity_type == 'text_link':
            escape_chars = r'\)'
        else:
            escape_chars = r'_*[]()~`>#+-=|{}.!'

        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    def send_telegram(self, message: str) -> bool:
        """
        Send a message to Telegram.

        Args:
            message (str): The message to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            data = {
                "chat_id": self.chat_id,
                "text": self.escape_markdown(message),
                "parse_mode": "MarkdownV2"
            }
            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(
                self.base_url, data=json.dumps(data), headers=headers)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending telegram message: {e}")
            return False

    def send_error(self, message: str) -> bool:
        """Send an error message with error emoji."""
        return self.send_telegram(f'{self.error} {message}')

    def send_success(self, message: str) -> bool:
        """Send a success message with success emoji."""
        return self.send_telegram(f'{self.success} {message}')

    def send_warning(self, message: str) -> bool:
        """Send a warning message with warning emoji."""
        return self.send_telegram(f'{self.warning} {message}')

    def send_message(self, message: str) -> bool:
        """Send a plain message without emoji."""
        return self.send_telegram(message)
