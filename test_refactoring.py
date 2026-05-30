import os
import sys
import asyncio
from dotenv import load_dotenv

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load env variables
load_dotenv()

from src.services.expertise import expertise_service
from src.services.message_assistant import (
    generate_message_assistant_response,
    answer_about_conversation,
    stream_answer_about_conversation,
)

async def test_expertise():
    print("Testing Expertise Service...")
    user_info = {
        "firstname": "Alex",
        "lastname": "Smith",
        "email": "alex.smith@example.com",
    }
    answers = {
        "role": "Frontend developer",
        "skills": "React, JavaScript, CSS",
        "interests": "UI design, web accessibility",
        "about": "Passionate about creating beautiful user interfaces.",
        "highlights": "Built a SaaS dashboard with React and Tailwind CSS.",
        "location": "San Francisco, CA",
    }
    
    # 1. Fresh Mode
    result = await expertise_service.generate_expertise(
        user_info=user_info,
        answers=answers,
        existing_expertise={},
        mode="fresh"
    )
    
    assert "expertise" in result
    assert "generation_source" in result
    expertise = result["expertise"]
    assert expertise["name"] == "Alex Smith"
    assert "React" in expertise["skills"]
    print("✅ Expertise Service Fresh Mode passed.")

    # 2. Refine Existing Mode
    existing = {
        "name": "Alex Smith",
        "description": "Frontend Developer",
        "experience": "React developer with 3 years of experience.",
        "skills": ["React", "JavaScript"],
        "projects": "Built dashboard.",
        "achievements": "Improved performance by 20%.",
        "interests": "Web dev",
        "aboutYou": "Developer",
        "details": {"email": "alex.smith@example.com", "address": "San Francisco"},
        "format": 1
    }
    result_refine = await expertise_service.generate_expertise(
        user_info=user_info,
        answers=answers,
        existing_expertise=existing,
        mode="refine_existing"
    )
    assert "expertise" in result_refine
    print("✅ Expertise Service Refine Mode passed.")


async def test_message_assistant():
    print("Testing Message Assistant Service...")
    
    # 1. Safety Block Test (e.g. blackmail or threat keywords)
    safety_result = generate_message_assistant_response(
        conversation=[],
        draft="I will blackmail them if they don't reply",
        tone="polite"
    )
    assert safety_result.get("error") == "safety_block"
    assert safety_result.get("routing_source") == "safety_guard"
    print("✅ Message Assistant Safety Guard passed.")
    
    # 2. Intro/Greeting Test
    greeting_result = answer_about_conversation(
        conversation=[],
        assistant_history=[],
        question="hello",
        tone="friendly"
    )
    try:
        assert greeting_result.get("mode") == "assistant_greeting"
    except AssertionError:
        print(f"FAILED GREETING RESULT: {greeting_result}")
        raise


    assert "Hello" in greeting_result["answer"]
    print("✅ Message Assistant Greeting Intent passed.")

    # 3. Intro/Capability Test
    capability_result = answer_about_conversation(
        conversation=[],
        assistant_history=[],
        question="What can you do?",
        tone="polite"
    )
    assert capability_result.get("mode") == "assistant_capabilities"
    print("✅ Message Assistant Capabilities Intent passed.")

    # 4. Contextual Response Test (mocking conversation history)
    conversation = [
        {"role": "other", "text": "Are you free for coffee tomorrow?", "kind": "text"},
        {"role": "me", "text": "Sure, what time?", "kind": "text"},
        {"role": "other", "text": "How about 10am at Starbucks?", "kind": "text"}
    ]
    
    replies = generate_message_assistant_response(
        conversation=conversation,
        draft="Yeah that works",
        tone="polite"
    )
    
    assert "top_reply" in replies
    assert "reply_suggestions" in replies
    assert len(replies["reply_suggestions"]) > 0
    print("✅ Message Assistant Contextual Reply Generation passed.")

    # 5. Streaming Answer Test
    print("Testing Streaming Response...")
    chunks = []
    async for chunk in stream_answer_about_conversation(
        conversation=conversation,
        assistant_history=[],
        question="What time did they suggest?",
        tone="polite"
    ):
        chunks.append(chunk)
    
    full_response = "".join(chunks)
    assert "10" in full_response or "starbucks" in full_response.lower()
    print("✅ Message Assistant Streaming Answer passed.")


async def main():
    try:
        await test_expertise()
        await test_message_assistant()
        print("\n🎉 ALL REFRACTORING TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
