"""
Test WhatsApp Alert - Standalone Script
This script tests the Twilio WhatsApp integration independently
"""
from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get credentials
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_NUMBER = os.getenv("TO_NUMBER")

print("=" * 60)
print("🔍 Testing Twilio WhatsApp Alert Configuration")
print("=" * 60)
print(f"TWILIO_SID: {TWILIO_SID[:10]}... (hidden)")
print(f"TWILIO_AUTH_TOKEN: {TWILIO_AUTH_TOKEN[:10]}... (hidden)")
print(f"FROM: {TWILIO_WHATSAPP_FROM}")
print(f"TO: {TO_NUMBER}")
print("=" * 60)

if not TWILIO_SID or not TWILIO_AUTH_TOKEN:
    print("❌ ERROR: Twilio credentials not found!")
    exit(1)

if not TO_NUMBER:
    print("❌ ERROR: TO_NUMBER not configured!")
    exit(1)

try:
    # Initialize Twilio client
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    print("✅ Twilio client initialized successfully")
    
    # Send test message
    print("\n📱 Sending test WhatsApp message...")
    message = client.messages.create(
        from_=TWILIO_WHATSAPP_FROM,
        body="🧪 *Test Alert from Pregnancy Risk API* 🧪\n\nThis is a test message to verify WhatsApp integration is working correctly.\n\nIf you receive this, the system is ready!",
        to=TO_NUMBER
    )
    
    print("=" * 60)
    print("✅ SUCCESS! WhatsApp message sent!")
    print(f"📋 Message SID: {message.sid}")
    print(f"📊 Status: {message.status}")
    print(f"📱 To: {message.to}")
    print(f"📤 From: {message.from_}")
    print("=" * 60)
    print("\n⏳ Check your WhatsApp now! You should receive the message shortly.")
    print("⚠️ NOTE: Make sure you've sent 'join <sandbox-code>' to the Twilio WhatsApp number first!")
    
except Exception as e:
    print("=" * 60)
    print("❌ FAILED to send WhatsApp message!")
    print(f"Error: {e}")
    print("=" * 60)
    print("\n💡 Troubleshooting tips:")
    print("1. Verify your Twilio credentials are correct")
    print("2. Make sure you've joined the Twilio WhatsApp sandbox")
    print("3. Send 'join <sandbox-code>' to +14155238886 on WhatsApp")
    print("4. Check if your Twilio account is active and has credits")
