"""
Crisis Support Message

This module provides a fixed, static crisis support message.
This message must NEVER change dynamically - it is a safety-critical constant.

The message is displayed to users when a crisis situation is detected.
It provides immediate resources and support information.
"""


def get_crisis_message() -> str:
    """
    Returns a fixed crisis support message in English.
    
    This message is static and must never be dynamically generated or modified.
    It provides users with immediate resources when a crisis is detected.
    
    Returns:
        A fixed string containing crisis support resources and guidance in English
    """
    return (
        "It sounds like you might not be feeling safe right now. I'm really glad you reached out.\n\n"
        "I'm not able to provide the kind of support you deserve in moments like this, but you don't have to go through it alone. You deserve help and support from trained professionals.\n\n"
        "Here are some trusted resources in Pakistan that you can reach out to immediately:\n\n"
        "Emergency / Immediate Safety:\n"
        "• Call 1122 for ambulance/medical help\n"
        "• Call 15 for police emergency\n\n"
        "Mental Health & Crisis Support:\n"
        "• Umang Helpline: 042 3576 5951\n"
        "• Taskeen Helpline: 0316 8275336\n"
        "• Rozan Counseling Helpline: 0311 7786264\n\n"
        "Additional Support Services:\n"
        "• National Youth Helpline: 0800 69457\n"
        "• Punjab Women's Toll-Free Helpline: 1043\n"
        "• Women & Children's Helpline: 1099\n\n"
        "You are not alone, and there are people who want to help you right now.\n\n"
        "Please prioritize your safety and reach out for professional support immediately."
    )


def get_crisis_message_urdu() -> str:
    """
    Returns a fixed crisis support message in Urdu.
    
    This message is static and must never be dynamically generated or modified.
    It provides users with immediate resources when a crisis is detected.
    
    Returns:
        A fixed string containing crisis support resources and guidance in Urdu
    """
    return (
        "ایسا لگتا ہے کہ آپ اس وقت محفوظ محسوس نہیں کر رہے ہیں۔ میں خوش ہوں کہ آپ نے مدد کے لیے رابطہ کیا۔\n\n"
    "میں ایسے لمحات میں وہ مدد فراہم نہیں کر سکتا جس کے آپ مستحق ہیں، لیکن آپ کو اکیلے اس صورتحال سے گزرنے کی ضرورت نہیں ہے۔ "
    "آپ کو تربیت یافتہ ماہرین سے مدد اور سپورٹ ملنی چاہیے۔\n\n"
    "پاکستان میں چند معتبر وسائل یہ ہیں جن سے آپ فوری طور پر رابطہ کر سکتے ہیں:\n\n"
    "ایمرجنسی / فوری حفاظت:\n"
    "• ایمبولینس یا طبی مدد کے لیے 1122 پر کال کریں\n"
    "• پولیس ایمرجنسی کے لیے 15 پر کال کریں\n\n"
    "ذہنی صحت اور بحران کی مدد:\n"
    "• اُمنگ ہیلپ لائن: 042 3576 5951\n"
    "• تسکین ہیلپ لائن: 0316 8275336\n"
    "• روزان کونسلنگ ہیلپ لائن: 0311 7786264\n\n"
    "اضافی مددگار خدمات:\n"
    "• نیشنل یوتھ ہیلپ لائن: 0800 69457\n"
    "• پنجاب ویمنز ٹول فری ہیلپ لائن: 1043\n"
    "• ویمن اینڈ چلڈرن ہیلپ لائن: 1099\n\n"
    "آپ اکیلے نہیں ہیں، اور ایسے لوگ موجود ہیں جو ابھی آپ کی مدد کرنا چاہتے ہیں۔\n\n"
    "براہ کرم اپنی حفاظت کو ترجیح دیں اور فوری طور پر پیشہ ورانہ مدد حاصل کریں۔"
    )

