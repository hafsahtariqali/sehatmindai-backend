"""
Email Service for sending OTP emails
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Email configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USERNAME)


class EmailService:
    """Service for sending emails via SMTP"""
    
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.smtp_username = SMTP_USERNAME
        self.smtp_password = SMTP_PASSWORD
        self.email_from = EMAIL_FROM
        
        if not self.smtp_username or not self.smtp_password:
            logger.warning(
                "SMTP credentials not configured. "
                "Set SMTP_USERNAME and SMTP_PASSWORD environment variables."
            )
    
    def send_otp_email(self, to_email: str, otp: str) -> bool:
        """
        Send OTP email to user
        
        Args:
            to_email: Recipient email address
            otp: 6-digit OTP code
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.smtp_username or not self.smtp_password:
            logger.error("Cannot send email: SMTP credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = "SehatMind - Password Reset OTP"
            msg['From'] = self.email_from
            msg['To'] = to_email
            
            # Create HTML email body
            html_body = f"""
            <html>
              <head></head>
              <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                  <div style="background-color: #1C477F; padding: 20px; border-radius: 10px 10px 0 0;">
                    <h1 style="color: white; margin: 0; font-size: 24px;">SehatMind</h1>
                  </div>
                  <div style="background-color: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                    <h2 style="color: #1C477F; margin-top: 0;">Password Reset Request</h2>
                    <p>You have requested to reset your password for your SehatMind account.</p>
                    <p>Please use the following OTP code to verify your identity:</p>
                    <div style="background-color: #4F8FC0; color: white; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                      <h1 style="margin: 0; font-size: 32px; letter-spacing: 5px;">{otp}</h1>
                    </div>
                    <p style="color: #666; font-size: 14px;">This code will expire in 10 minutes.</p>
                    <p style="color: #666; font-size: 14px;">If you did not request this password reset, please ignore this email.</p>
                    <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">
                    <p style="color: #999; font-size: 12px; text-align: center;">
                      This is an automated message from SehatMind. Please do not reply to this email.
                    </p>
                  </div>
                </div>
              </body>
            </html>
            """
            
            # Create plain text version
            text_body = f"""
SehatMind - Password Reset OTP

You have requested to reset your password for your SehatMind account.

Your OTP code is: {otp}

This code will expire in 10 minutes.

If you did not request this password reset, please ignore this email.

---
This is an automated message from SehatMind. Please do not reply to this email.
            """
            
            # Attach both versions
            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"OTP email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send OTP email to {to_email}: {str(e)}")
            return False

