from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import voice_assistant
from livekit.plugins import openai, silero

load_dotenv()


# Define the agent with Azure real-time model
agent = openai.realtime.RealtimeModel.with_azure(
    azure_deployment=os.getenv("MODEL_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    voice="alloy",
    temperature=0.8,
    instructions="You are a helpful assistant",
    turn_detection=openai.realtime.ServerVadOptions(
        threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
    ),
)


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Create a voice assistant agent context
    assistant_ctx = voice_assistant.VoicePipelineAgent(
        llm=agent,
        vad=silero.VAD.load(),
        # Removed turn_detection to avoid initialization issues
    )

    # Start the voice assistant
    await assistant_ctx.start(
        room=ctx.room,
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
