import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.plugins import assemblyai, inworld, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Increase verbosity of LiveKit Agents
logging.getLogger("livekit").setLevel(logging.DEBUG)
logging.getLogger("livekit.rtc").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)

# STT-specific logs
logging.getLogger("livekit.agents.stt").setLevel(logging.DEBUG)
logging.getLogger("livekit.plugins.assemblyai").setLevel(logging.DEBUG)

# Audio pipeline debugging
logging.getLogger("livekit.audio").setLevel(logging.DEBUG)

# Websocket/HTTP layer
logging.getLogger("aiohttp").setLevel(logging.DEBUG)
logging.getLogger("websockets").setLevel(logging.DEBUG)

logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)

load_dotenv(".env.local")


# Room + Track events
async def on_track_subscribed(participant, track):
    logger.debug(
        f"[TRACK SUBSCRIBED] participant={participant.identity}, "
        f"track={track.sid}, kind={track.kind}"
    )

async def on_track_unsubscribed(participant, track):
    logger.warning(
        f"[TRACK UNSUBSCRIBED] participant={participant.identity}, "
        f"track={track.sid}, kind={track.kind}"
    )

async def on_participant_joined(participant):
    logger.debug(f"[PARTICIPANT JOINED] {participant.identity}")

async def on_participant_left(participant):
    logger.debug(f"[PARTICIPANT LEFT] {participant.identity}")

# Audio frame debugging
async def on_audio_frame(frame):
    logger.debug(
        f"[AUDIO FRAME] ts={frame.timestamp}, samples={len(frame.data)}, "
        f"sample_rate={frame.sample_rate}"
    )

# Agent session lifecycle
async def on_session_start(session: AgentSession):
    logger.info(f"[SESSION START] job={session.job.id}, room={session.room.name}")

async def on_session_end(session: AgentSession):
    logger.info(f"[SESSION END] job={session.job.id}, room={session.room.name}")

# STT lifecycle debugging
async def on_stt_start():
    logger.info("[STT START] Opening realtime websocket to AssemblyAI")

async def on_stt_error(err):
    logger.error(f"[STT ERROR] {err}")

async def on_stt_reconnect(attempt, delay):
    logger.warning(f"[STT RECONNECT] attempt={attempt}, retry_in={delay}s")

async def on_stt_close():
    logger.info("[STT CLOSED] STT websocket closed")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        stt=assemblyai.STT(model="universal-streaming", debug=True),
        
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        # llm=inference.LLM(model="openai/gpt-4.1-mini"),
        llm=openai.LLM(model="gpt-5-nano"),
        
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        # tts=inference.TTS(
        #     model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        # ),
        tts=inworld.TTS(model="inworld-tts-1", voice="Dennis"),

        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Attach agent session hooks
    session.on_session_start = on_session_start
    session.on_session_end = on_session_end

    # Attach STT lifecycle hooks if supported by plugin
    if hasattr(session.stt, "on"):
        session.stt.on("start", on_stt_start)
        session.stt.on("error", on_stt_error)
        session.stt.on("reconnect", on_stt_reconnect)
        session.stt.on("close", on_stt_close)

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    io = await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=None
            ),
        ),
    )

    # -----------------------
    # ATTACH ROOM EVENT HOOKS
    # -----------------------
    io.on("track_subscribed", on_track_subscribed)
    io.on("track_unsubscribed", on_track_unsubscribed)
    io.on("participant_joined", on_participant_joined)
    io.on("participant_left", on_participant_left)
    io.on("audio_frame", on_audio_frame)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)