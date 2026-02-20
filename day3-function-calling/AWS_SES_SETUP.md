# AWS SES Setup Guide for Nimbus

## Step 1: Get AWS Credentials

### Option A: If you already have an AWS account
1. Go to AWS Console → IAM → Users → Your user → Security credentials
2. Create access key → CLI
3. Save the Access Key ID and Secret Access Key

### Option B: Create a new AWS account (free tier)
1. Go to https://aws.amazon.com/free/
2. Create account (requires credit card, but we'll stay in free tier)
3. After signup, go to IAM and create an access key

## Step 2: Verify Your Email in SES

AWS SES starts in "sandbox mode" - you can only send to verified emails.

1. Go to AWS Console → Amazon SES → Verified identities
2. Click "Create identity"
3. Choose "Email address"
4. Enter your email (e.g., your-email@example.com)
5. Click "Create identity"
6. **Check your email** and click the verification link
7. Wait for status to show "Verified"

## Step 3: Set Environment Variables

```bash
# Add to ~/.zshrc
export AWS_ACCESS_KEY_ID="your_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"
export AWS_REGION="us-east-1"  # or your preferred region
export SES_FROM_EMAIL="your-verified-email@example.com"  # your verified email

# Reload shell
source ~/.zshrc
```

## Step 4: Test SES Access

Run this to verify your credentials work:

```bash
python3 ~/workspace/llm-learning/day3-function-calling/test_ses.py
```

## Step 5: Restart Nimbus

```bash
# Stop current Nimbus (Ctrl+C if running)
cd ~/workspace/nimbus
go run cmd/gateway/main.go
```

## Step 6: Test with LLM

```bash
python3 ~/workspace/llm-learning/day3-function-calling/nimbus_integration.py \
  --request "Send an email to test@example.com with subject 'Test from Nimbus AI' and body 'This is a real email!'"
```

## Troubleshooting

### "Email address not verified"
- Go to SES console, verify your FROM email first
- In sandbox mode, verify RECIPIENT emails too
- To send to any email, request production access (takes 24hrs)

### "Access Denied"
- Your IAM user needs `ses:SendEmail` permission
- Add policy: `AmazonSESFullAccess` (or create custom policy)

### "Rate limit exceeded"
- Free tier: 200 emails/day, 1 email/second
- Add delays between sends if needed

## What You Need

**Minimum for testing:**
- AWS Access Key + Secret
- One verified email address (your own)
- Send emails FROM and TO the same verified address

**For production:**
- Request SES production access (removes sandbox limits)
- Verify your domain (not just email)
- Set up DKIM/SPF records
