"""Conversational Chatbot with Memory using DXMT.

This example demonstrates building an intelligent chatbot that:
- Retrieves relevant context from a knowledge base (DXMT vector store)
- Maintains conversation history for context-aware responses
- Provides natural, contextual answers using GPT

Prerequisites:
- DXMT server running on localhost:9999
- OpenAI API key set in environment: export OPENAI_API_KEY='sk-...'

Install dependencies:
    pip install langchain-dxmt langchain-openai langchain langchain-community langchain-classic

Use case:
    Customer support chatbot, personal assistant, or interactive documentation
    that remembers conversation context and retrieves relevant information.
"""

import os
from typing import List

# `langchain_classic` is imported inside `main`, not here: it is a separate package,
# and without it this example died at import time before it could say so.
from langchain_core.documents import Document
from embedding import default_embeddings, embed_dimension, needs_an_llm

# `langchain_openai` is imported where it is used, not here: without the
# package this example died at import time, before it could say what it needed.

from langchain_dxmt import DXMTVectorDB


def create_product_knowledge_base() -> List[Document]:
    """Create a knowledge base about a fictional smart home product."""
    docs = [
        Document(
            page_content="""
            The SmartHome Hub is our flagship product that connects all your smart
            devices. It supports WiFi, Zigbee, and Z-Wave protocols. The hub can
            control up to 200 devices simultaneously and works with Alexa, Google
            Assistant, and Apple HomeKit.
            """,
            metadata={"category": "product", "topic": "overview"}
        ),
        Document(
            page_content="""
            Setup is simple: 1) Download the SmartHome app, 2) Plug in the hub,
            3) Follow in-app instructions to connect to WiFi, 4) Add devices by
            scanning QR codes. The entire process takes about 10 minutes. The hub
            will automatically discover nearby devices.
            """,
            metadata={"category": "setup", "topic": "installation"}
        ),
        Document(
            page_content="""
            To add a new device: Open the app, tap "Add Device", select device type
            (light, lock, thermostat, camera, etc.), and follow pairing instructions.
            Most devices pair in under 30 seconds. If pairing fails, ensure the
            device is in pairing mode (check device manual) and within 30 feet of hub.
            """,
            metadata={"category": "setup", "topic": "device_pairing"}
        ),
        Document(
            page_content="""
            Create automations to make your home smarter: "Good Morning" routine
            can turn on lights, adjust thermostat, and start coffee maker. "Leaving
            Home" can lock doors, turn off lights, and arm security. Set schedules
            or use triggers like time, location, or device state.
            """,
            metadata={"category": "features", "topic": "automations"}
        ),
        Document(
            page_content="""
            If the hub is offline: 1) Check power connection, 2) Verify WiFi is
            working, 3) Restart hub by unplugging for 10 seconds, 4) Check router
            settings (hub needs 2.4GHz WiFi). If still offline, factory reset by
            holding reset button for 10 seconds. Hub LED should blink blue when online.
            """,
            metadata={"category": "troubleshooting", "topic": "connectivity"}
        ),
        Document(
            page_content="""
            Device not responding? Try: 1) Check device battery/power, 2) Ensure
            device is within range (30 feet), 3) Remove and re-add device in app,
            4) Update device firmware via app settings. For persistent issues,
            contact support with device model and error message.
            """,
            metadata={"category": "troubleshooting", "topic": "device_issues"}
        ),
        Document(
            page_content="""
            The hub updates automatically overnight. To manually update: Go to
            Settings > System > Check for Updates. Current version is 2.5.3.
            Updates include security patches, new device support, and performance
            improvements. Hub will restart during update (takes 2-3 minutes).
            """,
            metadata={"category": "maintenance", "topic": "updates"}
        ),
        Document(
            page_content="""
            We offer a 2-year warranty covering manufacturing defects. If your hub
            malfunctions within warranty period, contact support for free replacement.
            Extended warranty available for $29/year. Accidental damage not covered.
            Keep your receipt as proof of purchase.
            """,
            metadata={"category": "support", "topic": "warranty"}
        ),
        Document(
            page_content="""
            Privacy and security: All data is encrypted end-to-end. Hub connects
            to cloud only for voice assistant integration and remote access. You
            can enable local-only mode in Settings > Privacy. We never sell your
            data. Two-factor authentication available for account protection.
            """,
            metadata={"category": "security", "topic": "privacy"}
        ),
        Document(
            page_content="""
            Voice control: Say "Hey Google, turn on living room lights" or "Alexa,
            set thermostat to 72". To enable: Link SmartHome account in Alexa/Google
            Home app. You can create custom voice commands in SmartHome app under
            Voice Settings. Supports English, Spanish, French, German, and Chinese.
            """,
            metadata={"category": "features", "topic": "voice_control"}
        ),
    ]
    return docs


def main():
    """Run the conversational chatbot example."""
    print("=" * 80)
    print("Conversational Chatbot with Memory")
    print("=" * 80)

    # Check API key
    if not needs_an_llm("this example"):
        return

    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory

    from langchain_openai import ChatOpenAI

    # 1. Initialize embeddings and vector store
    print("\n1. Initializing knowledge base...")
    embeddings = default_embeddings()

    vectorstore = DXMTVectorDB(
        table_name="chatbot_kb",
        embedding=embeddings,
        dimension=embed_dimension(embeddings),
        host="localhost",
        port=9999,
        distance_metric="cosine",
        index_type="hnsw",
    )
    print("   ✅ Vector store initialized")

    # 2. Load knowledge base
    print("\n2. Loading product documentation...")
    docs = create_product_knowledge_base()
    ids = vectorstore.add_documents(docs)
    print(f"   ✅ Loaded {len(ids)} documents into knowledge base")

    # 3. Initialize LLM
    print("\n3. Initializing ChatGPT...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,  # Slightly creative for natural conversation
    )
    print("   ✅ LLM initialized")

    # 4. Create conversation memory
    print("\n4. Setting up conversation memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # Store only the answer, not source docs
    )
    print("   ✅ Memory initialized")

    # 5. Create conversational retrieval chain
    print("\n5. Creating conversational chain...")
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 relevant docs
        ),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    print("   ✅ Chatbot ready!")

    # 6. Demonstration conversation
    print("\n" + "=" * 80)
    print("Demo Conversation (Shows memory and context understanding)")
    print("=" * 80)

    demo_questions = [
        "Hi! What is the SmartHome Hub?",
        "How do I set it up?",  # "it" refers to hub from previous question
        "What if my hub goes offline?",  # Continues context
        "Can I use voice commands?",
        "What languages are supported?",  # Follows up on voice commands
    ]

    print("\n💡 Notice how the chatbot:")
    print("   - Remembers previous questions and answers")
    print("   - Understands pronouns like 'it' referring to previous topics")
    print("   - Provides contextual follow-up answers\n")

    for question in demo_questions:
        print(f"\n{'─' * 80}")
        print(f"👤 User: {question}")
        print(f"{'─' * 80}")

        # Get response
        result = chatbot.invoke({"question": question})
        answer = result["answer"]
        sources = result["source_documents"]

        print(f"\n🤖 Assistant: {answer}")

        # Show sources (optional, comment out for cleaner output)
        if sources:
            print(f"\n   📚 Sources: {len(sources)} docs", end="")
            categories = set(doc.metadata.get("category", "N/A") for doc in sources)
            print(f" ({', '.join(categories)})")

    # 7. Show conversation history
    print("\n" + "=" * 80)
    print("Conversation History")
    print("=" * 80)

    # Get the conversation history from memory
    chat_history = memory.load_memory_variables({})["chat_history"]
    print(f"\nTotal messages in memory: {len(chat_history)}")
    print("\nSummary:")
    for i, msg in enumerate(chat_history):
        role = "User" if msg.type == "human" else "Assistant"
        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"   {i+1}. {role}: {content_preview}")

    # 8. Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode - Try asking follow-up questions!")
    print("=" * 80)
    print("💡 Examples:")
    print("   - 'How do I add a new light?'")
    print("   - 'What if pairing fails?' (follow-up)")
    print("   - 'Tell me about automations'")
    print("   - 'Can you explain that again?' (uses memory)")
    print("\nType 'quit' to exit, 'history' to see conversation, 'clear' to reset\n")

    try:
        while True:
            user_input = input("👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using SmartHome support! Goodbye!")
                break

            if user_input.lower() == "clear":
                memory.clear()
                print("   🧹 Conversation history cleared!")
                continue

            if user_input.lower() == "history":
                history = memory.load_memory_variables({})["chat_history"]
                print(f"\n   📜 Conversation has {len(history)} messages")
                for i, msg in enumerate(history[-10:]):  # Show last 10
                    role = "You" if msg.type == "human" else "Bot"
                    print(f"   {i+1}. {role}: {msg.content[:80]}...")
                print()
                continue

            if not user_input:
                continue

            # Get response
            result = chatbot.invoke({"question": user_input})
            answer = result["answer"]

            print(f"\n🤖 Assistant: {answer}\n")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

    # 9. Cleanup
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)

    print("\nDeleting knowledge base...")
    vectorstore.delete(ids)
    print("✅ Documents deleted")

    print("\n" + "=" * 80)
    print("💡 Key Features Demonstrated")
    print("=" * 80)
    print("""
    ✅ Conversation Memory:
       - Chatbot remembers previous questions and answers
       - Understands context and pronouns ("it", "that", etc.)
       - Enables natural follow-up questions

    ✅ Knowledge Retrieval:
       - Searches knowledge base for relevant information
       - Combines retrieved context with conversation history
       - Provides accurate, grounded answers

    ✅ Natural Language Understanding:
       - Interprets user intent from conversational text
       - Handles variations in phrasing
       - Maintains conversation flow

    💡 Production Tips:
    1. Use ConversationSummaryMemory for long conversations (more efficient)
    2. Add conversation metadata (user_id, session_id) for multi-user support
    3. Implement conversation persistence (Redis, database)
    4. Add intent classification for better routing
    5. Use streaming for real-time responses
    6. Add feedback collection to improve knowledge base
    """)


if __name__ == "__main__":
    main()
