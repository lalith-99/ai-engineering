"""
Quick test script to verify AWS SES credentials are working.
Run this BEFORE starting Nimbus to catch config issues early.
"""

import os
import sys

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("boto3 not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "--quiet"])
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError


def test_ses_config():
    """Test AWS SES configuration and credentials."""
    
    print("Checking AWS SES Configuration...\n")
    
    # Check environment variables
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")
    from_email = os.getenv("SES_FROM_EMAIL")
    
    print(f"  AWS_ACCESS_KEY_ID: {'Set' if access_key else 'MISSING'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'Set' if secret_key else 'MISSING'}")
    print(f"  AWS_REGION: {region}")
    print(f"  SES_FROM_EMAIL: {from_email if from_email else 'MISSING'}\n")
    
    if not access_key or not secret_key:
        print("ERROR: AWS credentials not found!")
        print("   Export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("   See AWS_SES_SETUP.md for instructions")
        return False
    
    if not from_email:
        print("Warning: SES_FROM_EMAIL not set, using default")
        from_email = "noreply@nimbus.local"
    
    # Test SES connection
    try:
        print("Connecting to AWS SES...")
        client = boto3.client('ses', region_name=region)
        
        # Get account send quota
        quota = client.get_send_quota()
        print(f"\nConnected to SES in {region}")
        print(f"   Max 24hr send: {quota['Max24HourSend']:.0f}")
        print(f"   Sent last 24hr: {quota['SentLast24Hours']:.0f}")
        print(f"   Max send rate: {quota['MaxSendRate']:.0f}/sec\n")
        
        # List verified email addresses
        print("Verified email identities:")
        identities = client.list_identities(IdentityType='EmailAddress')
        
        if not identities['Identities']:
            print("   No verified emails found!")
            print("   Go to AWS SES Console and verify an email address")
            print("   See AWS_SES_SETUP.md Step 2")
            return False
        
        for email in identities['Identities']:
            # Check verification status
            attrs = client.get_identity_verification_attributes(Identities=[email])
            status = attrs['VerificationAttributes'].get(email, {}).get('VerificationStatus', 'Unknown')
            
            icon = "[OK]" if status == "Success" else "[PENDING]" if status == "Pending" else "[FAIL]"
            print(f"   {icon} {email} ({status})")
        
        print()
        
        # Check if FROM email is verified
        if from_email not in identities['Identities']:
            print(f"Warning: FROM email '{from_email}' is NOT verified!")
            print("   Verify this email in SES Console first")
            return False
        
        print(f"FROM email '{from_email}' is verified!")
        
        # Check sandbox status
        print("\nAccount Status:")
        try:
            account_details = client.get_account_sending_enabled()
            if account_details['Enabled']:
                print("   Sending enabled")
            else:
                print("   Sending disabled")
                return False
        except:
            pass
        
        print("   Note: You're likely in SANDBOX mode")
        print("   → Can only send TO verified emails")
        print("   → Request production access to send to anyone")
        
        print("\n" + "="*60)
        print("AWS SES is configured correctly!")
        print("="*60)
        print("\nNext steps:")
        print("1. Restart Nimbus: cd ~/workspace/nimbus && go run cmd/gateway/main.go")
        print("2. Test email: python3 ~/workspace/llm-learning/day3-function-calling/nimbus_integration.py")
        
        return True
        
    except NoCredentialsError:
        print("ERROR: AWS credentials are invalid or expired")
        print("   Check your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidClientTokenId':
            print("ERROR: Invalid AWS Access Key ID")
        elif error_code == 'SignatureDoesNotMatch':
            print("ERROR: Invalid AWS Secret Access Key")
        else:
            print(f"ERROR: AWS Error: {e}")
        return False
        
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_ses_config()
    sys.exit(0 if success else 1)
