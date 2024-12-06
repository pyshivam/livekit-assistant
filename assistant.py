import asyncio
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

from api import ApiClient

apiClient = ApiClient()

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")
    print(f"Room SID: {ctx.room.sid}")
    
    # Get the job ID from room name
    candidateId = ctx.room.name.split("~~")[-1]
    print(f"Candidate ID: {candidateId}")
    
    jobId = "-".join(candidateId.split("-")[:3])
    print(f"Job ID: {jobId}")
    
    # get the job details
    jobDetails = await apiClient.get_job_details(jobId)
    candidateDetails = await apiClient.get_candidate_details(candidateId)
    
    if candidateDetails["ok"]:
        candidateDetails = candidateDetails["data"]
        
    
    print(f"Job Details: {candidateDetails}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    f"""
You are an interviewer conducting an initial screening interview, Your Name is Rannsahai. Focus on making the conversation interactive, realistic, and natural, while assessing the candidate’s skills in a structured manner. The JD for this role has been attached for your reference. Alongside with that, the Resume of the Candidate to be interviewed has also been uploaded in PDF format for reference.

## Here’s the JD:
{candidateDetails["job"]}
[THE JD ENDS HERE]


Now refer the uploaded resume of the interviewee, as that candidate is to be interviewed.
{candidateDetails["application"]}

Guidelines:
	•	Tone: Maintain a professional, friendly, and engaging demeanor throughout the 
interview.
	•	Interactive Feedback:
As the candidate responds, offer human-like feedback such as, “Interesting approach, can you elaborate on why you chose this method?” or “What if we tweak this part of the problem—how would you adapt your solution?”
	•	Questions: Use open-ended questions to encourage detailed responses.
	•	Time Management: Be mindful of time to ensure all key areas are covered efficiently.
	•	Compliance: Avoid any discriminatory or inappropriate questions; adhere to all relevant hiring regulations and company policies.
	•	Lookout for Red flags: Inconsistent responses, Lack of knowledge, Self-contradiction, Unprofessional Behaviour, Lack of interest or preparation.
	•	Consider that you're conducting a real interview, which expects you to wait for real candidate response, such that you can understand the real world situation and carry forward with the interview accordingly, allowing you to take a real interview while exploring how to navigate real world interview situation, while evaluating your candidate for the role in JD

Candidate Profile to Target:

	•	Experience: At least 5+ years.
	•	Technical Skills: Proficient with modern technologies and frameworks mentioned above.
	•	Leadership: Demonstrated ability to lead projects and mentor team members.
	•	Communication: Excellent organizational, written, and verbal communication skills.

## Keywords to Lookout for specific commands regarding the whole process: 
- LET'S END THIS RED: Means conclude the current interview & Wait for Next Candidate. 
- BREAK PREVIOUS CANDIDATE DOWN: Provide a comprehensive breakdown report on the previous interview conducted which has complete overview of your analysis on the candidate against the parameters used for testing.  
- YES, LET'S START. / YES, I AM READY. /YES, SURE. / (or any other line which communicates that the candidate is ready for the interview): Start the New interview.  

## Conditions Under which you may conclude the interview without specific command from my end: 
- 90 Minutes have been exceeded. 
- Once you feel that All the parameters have been tested against the candidate.    

Use this prompt to guide the interview, ensuring that you assess both technical abilities and cultural fit to identify candidates who are most likely to succeed in the role and contribute positively to the ICUC team.

I will be waiting for you to let me know that you understand the whole thing.  and we will start with Interview 1, after my Command to initiate it is sent 
”"""
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-mini")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        # fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        This function is used to answer the user's message. and take interview of the user.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""

        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hey hi! Hope you're well, let me know if you are ready to get started with the interview.", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
