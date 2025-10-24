#!/bin/bash

USER="Eren"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🔍 Checking for explicit Deny policies for user: $USER (Account: $ACCOUNT_ID)"
echo "==============================================================="

# 1️⃣ List policies directly attached to the user
echo -e "\n➡️  Directly attached policies:"
USER_POLICIES=$(aws iam list-attached-user-policies --user-name $USER --query "AttachedPolicies[].PolicyArn" --output text)
for POLICY_ARN in $USER_POLICIES; do
  echo "  🔸 Checking $POLICY_ARN"
  POLICY_DOC=$(aws iam get-policy-version --policy-arn $POLICY_ARN \
      --version-id $(aws iam get-policy --policy-arn $POLICY_ARN --query 'Policy.DefaultVersionId' --output text) \
      --query 'PolicyVersion.Document' --output json)
  echo "$POLICY_DOC" | grep -i '"Effect": *"Deny"' && echo "❌ Found Deny in $POLICY_ARN" || echo "✅ No Deny found"
done

# 2️⃣ Check inline policies attached to the user
echo -e "\n➡️  Inline policies attached to user:"
INLINE_POLICIES=$(aws iam list-user-policies --user-name $USER --query "PolicyNames[]" --output text)
for POLICY_NAME in $INLINE_POLICIES; do
  POLICY_DOC=$(aws iam get-user-policy --user-name $USER --policy-name $POLICY_NAME --query "PolicyDocument" --output json)
  echo "$POLICY_DOC" | grep -i '"Effect": *"Deny"' && echo "❌ Found Deny in inline policy: $POLICY_NAME" || echo "✅ No Deny found in $POLICY_NAME"
done

# 3️⃣ Check groups the user belongs to
echo -e "\n➡️  Checking groups for user:"
GROUPS=$(aws iam list-groups-for-user --user-name $USER --query "Groups[].GroupName" --output text)
for GROUP in $GROUPS; do
  echo "  🔹 Group: $GROUP"
  GROUP_POLICIES=$(aws iam list-attached-group-policies --group-name $GROUP --query "AttachedPolicies[].PolicyArn" --output text)
  for POLICY_ARN in $GROUP_POLICIES; do
    echo "    🔸 Checking $POLICY_ARN"
    POLICY_DOC=$(aws iam get-policy-version --policy-arn $POLICY_ARN \
        --version-id $(aws iam get-policy --policy-arn $POLICY_ARN --query 'Policy.DefaultVersionId' --output text) \
        --query 'PolicyVersion.Document' --output json)
    echo "$POLICY_DOC" | grep -i '"Effect": *"Deny"' && echo "    ❌ Found Deny in $POLICY_ARN" || echo "    ✅ No Deny found"
  done

  INLINE_GROUP_POLICIES=$(aws iam list-group-policies --group-name $GROUP --query "PolicyNames[]" --output text)
  for POLICY_NAME in $INLINE_GROUP_POLICIES; do
    POLICY_DOC=$(aws iam get-group-policy --group-name $GROUP --policy-name $POLICY_NAME --query "PolicyDocument" --output json)
    echo "$POLICY_DOC" | grep -i '"Effect": *"Deny"' && echo "    ❌ Found Deny in inline group policy: $POLICY_NAME" || echo "    ✅ No Deny found in $POLICY_NAME"
  done
done

# 4️⃣ Check if a Permissions Boundary is set
echo -e "\n➡️  Checking permissions boundary:"
BOUNDARY_ARN=$(aws iam get-user --user-name $USER --query "User.PermissionsBoundary.PermissionsBoundaryArn" --output text 2>/dev/null)
if [[ "$BOUNDARY_ARN" != "None" && "$BOUNDARY_ARN" != "" ]]; then
  echo "  🔸 Found boundary: $BOUNDARY_ARN"
  POLICY_DOC=$(aws iam get-policy-version --policy-arn $BOUNDARY_ARN \
      --version-id $(aws iam get-policy --policy-arn $BOUNDARY_ARN --query 'Policy.DefaultVersionId' --output text) \
      --query 'PolicyVersion.Document' --output json)
  echo "$POLICY_DOC" | grep -i '"Effect": *"Deny"' && echo "  ❌ Found Deny in permissions boundary" || echo "  ✅ No Deny found in permissions boundary"
else
  echo "  ✅ No permissions boundary set."
fi

echo -e "\n✅ Check complete!"
