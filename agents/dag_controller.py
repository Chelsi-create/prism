import re
import json

class DAGController:
    def __init__(self, agents, sum_agent):
        self.agents = agents
        self.sum_agent = sum_agent

        self.image_agent = self.agents[0]
        self.text_agent = self.agents[1]
        self.general_agent = self.agents[-1]

    def follow_dag(self, main_question, texts, images, reasoning_dag):
        previous_answers = {}
        all_messages = ""

        for parent, child in reasoning_dag:
            if parent in previous_answers:
                current_query = child
                placeholders = re.findall(r'<(A\d+\.\d+)>', current_query)
                for placeholder in placeholders:
                    if placeholder in previous_answers:
                        current_query = current_query.replace(f"<{placeholder}>", previous_answers[placeholder])
            else:
                current_query = parent

            agent = self.select_agent(texts, images)

            generated_ans, agent_message = agent.predict(current_query, texts=texts, images=images, with_sys_prompt=True)
            all_messages += f"Sub-question: {current_query}\n"
            all_messages += f"Answer: {generated_ans}\n\n"

            # Step 4: Save answer
            answer_key_match = re.search(r'(Q\d+\.\d+)', child)
            if answer_key_match:
                answer_key = "A" + answer_key_match.group(1)[1:]  # Example: Q1.2 -> A1.2
                previous_answers[answer_key] = generated_ans

        # Step 5: Summarize final answer using sum_agent
        final_ans, final_messages = self.sum_agent.predict(all_messages)

        # Step 6: Clean final answer
        try:
            final_ans_dict = json.loads(final_ans)
            final_ans = final_ans_dict.get("Answer", final_ans)
        except:
            pass

        return final_ans, all_messages

    def select_agent(self, texts, images):
        if texts and not images:
            return self.text_agent
        elif images and not texts:
            return self.image_agent
        else:
            return self.general_agent
