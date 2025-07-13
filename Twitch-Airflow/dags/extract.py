import os
import requests
import asyncio, time
import pandas as pd
from datetime import datetime, timedelta, timezone
from twitchio.ext import commands
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TWITCH_IRC_TOKEN = os.getenv("TWITCH_IRC_TOKEN")


def get_access_token():
    url = "https://id.twitch.tv/oauth2/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }
    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json()["access_token"]


twitch_oauth_token = get_access_token()

HEADERS = {"Client-ID": CLIENT_ID, "Authorization": f"Bearer {twitch_oauth_token}"}


def get_streamers():
    try:
        collected = []
        seen_ids = set()
        cursor = None
        fetch_count = 100

        while len(collected) < fetch_count:
            url = "https://api.twitch.tv/helix/streams"
            params = {"first": fetch_count}
            if cursor:
                params["after"] = cursor

            resp = requests.get(url, headers=HEADERS, params=params)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 1))
                print(f"⏳ Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()
            streams = data.get("data", [])
            cursor = data.get("pagination", {}).get("cursor")

            if not streams:
                break

            for stream in streams:
                user_id = stream["user_id"]
                if user_id in seen_ids:
                    continue

                collected.append(
                    {
                        "id": user_id,
                        "login": stream["user_login"],
                        "display_name": stream["user_name"],
                        "game_name": stream.get("game_name"),
                        "title": stream.get("title"),
                        "viewer_count": stream.get("viewer_count"),
                        "language": stream.get("language"),
                        "tags": (
                            ", ".join(stream.get("tags", []))
                            if stream.get("tags")
                            else None
                        ),
                    }
                )

                seen_ids.add(user_id)
                if len(collected) >= fetch_count:
                    break

            if not cursor:
                break

        df = pd.DataFrame(collected)
        return df
    except Exception as e:
        print(e)


class TwitchChatBot(commands.Bot):
    def __init__(self, channels):
        super().__init__(token=TWITCH_IRC_TOKEN, prefix="!")
        self.channels_to_join = channels
        self.run_time = timedelta(minutes=5)
        self._stopping = False
        self.ready_to_collect = False
        self.chat_log = []

    async def event_ready(self):
        print(f"Connected as {self.nick}")
        for channel in self.channels_to_join:
            for attempt in range(3):
                try:
                    await self.join_channels([channel])
                    print(f"Joined {channel}")
                    await asyncio.sleep(0.5)
                    break
                except Exception as e:
                    print(f"Join error for {channel}: {e}")
                    await asyncio.sleep(2)
        self.start_time = datetime.now(timezone.utc)
        self.ready_to_collect = True

    async def event_message(self, message):
        if message.echo or not self.ready_to_collect:
            return

        print(f"[{message.channel.name}] {message.author.name}: {message.content}")

        chat_data = {
            "message_id": message.id,
            "channel": message.channel.name,
            "message": message.content,
            "user_name": message.author.name,
            "user_id": message.author.id,
            "user_display_name": message.author.display_name,
            "user_is_mod": message.author.is_mod,
            "user_is_subscriber": message.author.is_subscriber,
            "user_is_vip": message.author.is_vip,
            "user_is_broadcaster": message.author.is_broadcaster,
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
            "first_message": message.tags.get("first-msg") == "1",
            "returning_chatter": message.tags.get("returning-chatter") == "1",
        }

        self.chat_log.append(chat_data)

        if (
            not self._stopping
            and datetime.now(timezone.utc) - self.start_time > self.run_time
        ):
            self._stopping = True
            await self.close()

    async def close(self):
        await super().close()
        print("IRC session closed.")


async def collect_chat_async(streamers):
    try:
        if streamers is None or streamers.empty:
            print("No streamers found.")
            return pd.DataFrame()

        channels = streamers["login"].str.lower().tolist()
        print(f"Collecting chat from channels: {channels}")

        bot = TwitchChatBot(channels)
        try:
            await asyncio.wait_for(bot.start(), timeout=120)
        except asyncio.TimeoutError:
            print("Chat collection timeout.")
        finally:
            chat_df = pd.DataFrame(bot.chat_log)
            return chat_df
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()
