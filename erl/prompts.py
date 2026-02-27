DEFAULT_REFLECTION_TEMPLATE: str = """\
You are analyzing a failed attempt at a task to help improve the next attempt.

## Original Task
{prompt}

## Your Previous Attempt
{attempt}

## Feedback Received
{feedback}

## Reward Score
{reward}

## Relevant Past Reflections
{memory}

Based on the above, write a concise reflection:

1. What specifically went wrong in the previous attempt?
2. What concrete changes should be made in the next attempt?
3. What strategy should be followed?

Be specific and actionable. Focus on what to change, not what was wrong in general.\
"""

DEFAULT_RETRY_TEMPLATE: str = """\
You are making a second attempt at a task, guided by a reflection on your previous attempt.

## Original Task
{prompt}

## Reflection and Improvement Plan
{reflection}

Now provide an improved response to the original task. Follow the improvement plan from the reflection.\
"""
