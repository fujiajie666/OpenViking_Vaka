import csv
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, 'result_timestamp')
SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts')


def load_csv_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    qa_files = [
        ('run1', os.path.join(RESULT_DIR, 'vaka_qa_result.csv')),
        ('run2', os.path.join(RESULT_DIR, 'vaka_qa_result_run2.csv')),
        ('run3', os.path.join(RESULT_DIR, 'vaka_qa_result_run3.csv')),
    ]

    qa_data = {}
    for run_name, path in qa_files:
        rows = load_csv_dict(path)
        qa_data[run_name] = {int(r['question_index']): r for r in rows}

    # 统计每个问题在3次运行中错误的次数
    wrong_counts = defaultdict(int)
    for run_name, rows in qa_data.items():
        for idx, row in rows.items():
            if row['result'] == 'WRONG':
                wrong_counts[idx] += 1

    # 情况1：模型不稳定（3次只错1次）
    unstable_indices = {idx for idx, cnt in wrong_counts.items() if cnt == 1}

    # 读取 stable_wrong_analysis
    stable_path = os.path.join(RESULT_DIR, 'stable_wrong_analysis.csv')
    stable_rows = load_csv_dict(stable_path)
    stable_by_index = {int(r['question_index']): r for r in stable_rows}

    # 情况2：ground truth 有问题
    unreasonable_indices = {
        int(r['question_index'])
        for r in stable_rows
        if r['standard_answer_reasonable'].startswith('UNREASONABLE')
    }

    # 情况3：原始对话质量低
    memory_wrong_indices = {
        int(r['question_index'])
        for r in stable_rows
        if r['memory_judge_result'] == 'WRONG'
    }

    # 所有需要输出的问题索引
    all_indices = sorted(unstable_indices | unreasonable_indices | memory_wrong_indices)

    output_path = os.path.join(RESULT_DIR, 'bad_questions.csv')

    # 构建输出列：stable_wrong_analysis 的列 + bad_type
    output_columns = list(stable_rows[0].keys()) + ['bad_type']

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()

        for idx in all_indices:
            bad_types = []
            if idx in unstable_indices:
                bad_types.append('模型不稳定')
            if idx in unreasonable_indices:
                bad_types.append('ground truth有问题')
            if idx in memory_wrong_indices:
                bad_types.append('输入的原始对话质量低')

            if idx in stable_by_index:
                # 情况2或3：直接复制 stable_wrong_analysis 的行
                row = dict(stable_by_index[idx])
            else:
                # 情况1：从3个QA结果文件中构建行
                run1_row = qa_data['run1'].get(idx, {})
                row = {
                    'question_index': idx,
                    'question': run1_row.get('question', ''),
                    'standard_answer': run1_row.get('standard_answer', ''),
                    'wrong_count': 1,
                    'total_runs': 3,
                    'result': 'WRONG',
                    'judge_mode': run1_row.get('judge_mode', ''),
                    'response_run1': qa_data['run1'].get(idx, {}).get('response', ''),
                    'reasoning_run1': qa_data['run1'].get(idx, {}).get('reasoning', ''),
                    'response_run2': qa_data['run2'].get(idx, {}).get('response', ''),
                    'reasoning_run2': qa_data['run2'].get(idx, {}).get('reasoning', ''),
                    'response_run3': qa_data['run3'].get(idx, {}).get('response', ''),
                    'reasoning_run3': qa_data['run3'].get(idx, {}).get('reasoning', ''),
                    'related_memory_text': '',
                    'standard_answer_reasonable': '',
                    'memory_generated_answer': '',
                    'memory_judge_result': '',
                    'memory_judge_reasoning': '',
                }

            row['bad_type'] = '、'.join(bad_types)
            writer.writerow(row)

    print(f"已生成 {output_path}")
    print(f"共输出 {len(all_indices)} 条异常问题：")
    print(f"  - 模型不稳定: {len(unstable_indices)} 条")
    print(f"  - ground truth有问题: {len(unreasonable_indices)} 条")
    print(f"  - 输入的原始对话质量低: {len(memory_wrong_indices)} 条")
    print(f"  - 重叠（同时属于多种类型）: {len(unstable_indices) + len(unreasonable_indices) + len(memory_wrong_indices) - len(all_indices)} 条")


if __name__ == '__main__':
    main()
