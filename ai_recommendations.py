# ai_recommendations.py
"""
Claude AI Integration for Personalised Retention Recommendations.
Calls the Anthropic API to generate contextual, customer-specific
retention strategies based on churn prediction data and SHAP explanations.
"""

import anthropic
import json


def build_prompt(customer_data, probability, shap_data=None):
    """Build a structured prompt for Claude."""

    risk_label = (
        "HIGH RISK" if probability > 0.7
        else "MEDIUM RISK" if probability > 0.4
        else "LOW RISK"
    )

    # Top churn drivers from SHAP
    drivers_text = ""
    if shap_data and shap_data.get('top_positive'):
        drivers = shap_data['top_positive'][:5]
        drivers_text = "\n".join(
            f"  - {name}: impact score {val:.3f}"
            for name, val in drivers
        )
        drivers_text = f"\nKey churn drivers identified by SHAP analysis:\n{drivers_text}"

    prompt = f"""You are a senior customer retention specialist at a telecom company.
Analyse the following customer profile and churn prediction, then write a concise, 
actionable retention strategy.

CUSTOMER PROFILE:
- Tenure: {customer_data.get('tenure', 'N/A')} months
- Contract: {customer_data.get('Contract', 'N/A')}
- Monthly Charges: ${customer_data.get('MonthlyCharges', 'N/A')}
- Total Charges: ${customer_data.get('TotalCharges', 'N/A')}
- Internet Service: {customer_data.get('InternetService', 'N/A')}
- Phone Service: {customer_data.get('PhoneService', 'N/A')}
- Online Security: {customer_data.get('OnlineSecurity', 'N/A')}
- Tech Support: {customer_data.get('TechSupport', 'N/A')}
- Payment Method: {customer_data.get('PaymentMethod', 'N/A')}
- Partner: {customer_data.get('Partner', 'N/A')}
- Senior Citizen: {'Yes' if customer_data.get('SeniorCitizen') == 1 else 'No'}
{drivers_text}

CHURN PREDICTION: {risk_label} ({probability:.1%} churn probability)

Write a retention strategy with these sections:
1. **Customer Assessment** (2 sentences: why this customer is at this risk level)
2. **Immediate Actions** (3 bullet points: specific actions for this week)
3. **Offer Recommendation** (1 specific offer tailored to their profile and charges)
4. **Long-term Retention** (2 bullet points: how to keep them engaged 6-12 months)

Be specific to THIS customer's profile. Keep total response under 250 words.
Use plain text with ** for bold. No generic advice."""

    return prompt


def get_ai_recommendation(customer_data, probability, shap_data=None,
                           api_key=None):
    """
    Call Claude API to get a personalised retention recommendation.

    Parameters
    ----------
    customer_data : dict  — customer feature values
    probability   : float — churn probability (0-1)
    shap_data     : dict  — SHAP output from shap_explainer.py (optional)
    api_key       : str   — Anthropic API key (reads ANTHROPIC_API_KEY env if None)

    Returns
    -------
    dict:
        success        : bool
        recommendation : str  (the AI text)
        error          : str  (if success=False)
    """
    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        prompt = build_prompt(customer_data, probability, shap_data)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )

        recommendation = message.content[0].text.strip()

        return {
            'success':        True,
            'recommendation': recommendation,
            'tokens_used':    message.usage.input_tokens + message.usage.output_tokens,
        }

    except anthropic.AuthenticationError:
        return {
            'success': False,
            'error':   'Invalid API key. Set ANTHROPIC_API_KEY environment variable.',
            'recommendation': _fallback_recommendation(customer_data, probability),
        }
    except anthropic.RateLimitError:
        return {
            'success': False,
            'error':   'API rate limit reached. Using fallback recommendation.',
            'recommendation': _fallback_recommendation(customer_data, probability),
        }
    except Exception as e:
        return {
            'success': False,
            'error':   str(e),
            'recommendation': _fallback_recommendation(customer_data, probability),
        }


def _fallback_recommendation(customer_data, probability):
    """Rule-based fallback when Claude API is unavailable."""
    contract   = customer_data.get('Contract', 'Month-to-month')
    monthly    = float(customer_data.get('MonthlyCharges', 0))
    tenure     = int(customer_data.get('tenure', 0))
    tech       = customer_data.get('TechSupport', 'No')
    security   = customer_data.get('OnlineSecurity', 'No')

    if probability > 0.7:
        offer = (
            f"Offer a 25% discount for 6 months (saving ${monthly*0.25:.0f}/month)"
            if monthly > 60 else
            "Offer a free upgrade to the next service tier for 3 months"
        )
        contract_action = (
            "Propose switching to an annual contract with first month free"
            if contract == 'Month-to-month' else
            "Acknowledge loyalty with an exclusive retention bonus"
        )
        return (
            f"**Customer Assessment**\nThis customer shows critical churn signals with "
            f"{probability:.0%} probability. Tenure of {tenure} months and current contract "
            f"type suggest immediate intervention is required.\n\n"
            f"**Immediate Actions**\n"
            f"• Call within 24 hours for a personalised retention conversation\n"
            f"• {offer}\n"
            f"• {contract_action}\n\n"
            f"**Offer Recommendation**\n"
            f"{'Add free Tech Support + Online Security for 3 months' if tech == 'No' or security == 'No' else 'Provide a complimentary service bundle upgrade'}\n\n"
            f"**Long-term Retention**\n"
            f"• Enrol in loyalty programme with monthly reward points\n"
            f"• Schedule 90-day check-in call to assess satisfaction"
        )
    elif probability > 0.4:
        return (
            f"**Customer Assessment**\nModerate churn risk detected ({probability:.0%}). "
            f"Customer has been with us {tenure} months — worth proactive engagement.\n\n"
            f"**Immediate Actions**\n"
            f"• Send personalised satisfaction survey with $10 bill credit for completion\n"
            f"• Offer a 10% loyalty discount on the next bill\n"
            f"• Recommend relevant service upgrades based on usage pattern\n\n"
            f"**Offer Recommendation**\n"
            f"Bundle discount: add {'Tech Support' if tech == 'No' else 'Streaming TV'} "
            f"for 50% off for 6 months\n\n"
            f"**Long-term Retention**\n"
            f"• Monthly engagement email with tips and exclusive offers\n"
            f"• Invite to beta programme for new features"
        )
    else:
        return (
            f"**Customer Assessment**\nLow churn risk ({probability:.0%}). "
            f"This customer shows healthy engagement with {tenure} months of tenure.\n\n"
            f"**Immediate Actions**\n"
            f"• Send a thank-you message acknowledging their loyalty\n"
            f"• Offer an upsell to a premium tier at a special rate\n"
            f"• Request a referral with a reward incentive\n\n"
            f"**Offer Recommendation**\n"
            f"Premium bundle upgrade with a 15% loyalty discount as a valued customer\n\n"
            f"**Long-term Retention**\n"
            f"• Enrol in VIP programme with priority support\n"
            f"• Early access to new products and features"
        )
