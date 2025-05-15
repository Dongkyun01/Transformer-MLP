import pandas as pd
import numpy as np
import pickle

# 1. 데이터 로딩
events = pd.read_csv(r'C:\Users\USER\Desktop\Transformer\Retailrocekt\events.csv')  # 경로 수정 필요

# 2. timestamp 변환 (밀리초 → datetime)
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# 3. 행동 인코딩
event_mapping = {'view': 0, 'addtocart': 1, 'transaction': 2}
events['event_encoded'] = events['event'].map(event_mapping)

# 4. 세션 그룹화
session_groups = events.groupby('visitorid')

# 4-1. 행동 시퀀스 생성
def create_event_sequences(groups):
    return pd.DataFrame([
        {'visitorid': vid, 'event_sequence': group.sort_values('timestamp')['event_encoded'].tolist()}
        for vid, group in groups
    ])

session_df = create_event_sequences(session_groups)

# 4-2. 추가 변수 생성
def create_additional_features(groups):
    return pd.DataFrame([
        {
            'visitorid': vid,
            'session_length': len(g),
            'event_diversity': g['event_encoded'].nunique(),
            'view_count': (g['event_encoded'] == 0).sum(),
            'addtocart_count': (g['event_encoded'] == 1).sum(),
            'transaction_count': (g['event_encoded'] == 2).sum(),
            'session_duration': (g['timestamp'].max() - g['timestamp'].min()).total_seconds(),
            'start_hour': g['timestamp'].min().hour,
            'start_weekday': g['timestamp'].min().weekday()
        }
        for vid, g in groups
    ])

additional_features_df = create_additional_features(session_groups)

# 5. 타겟 생성 (세션 간 시간 차이 사용)
session_last_time = session_groups['timestamp'].max().reset_index().sort_values('timestamp').reset_index(drop=True)
time_to_next = session_last_time['timestamp'].shift(-1) - session_last_time['timestamp']
time_to_next = time_to_next.dt.total_seconds()

# 마지막 방문자 제거
session_last_time = session_last_time.iloc[:-1]
time_to_next = time_to_next.iloc[:-1]

# 6. Align
session_df = session_df.set_index('visitorid').loc[session_last_time['visitorid']].reset_index()
additional_features_df = additional_features_df.set_index('visitorid').loc[session_last_time['visitorid']].reset_index()

# 7. 최종 변수 정의
session_event_sequences = session_df['event_sequence']
additional_features = additional_features_df.drop(columns=['visitorid'])
time_to_next_session = time_to_next

# 8. 저장
save_obj = {
    'session_event_sequences': session_event_sequences,
    'time_to_next_session': time_to_next_session,
    'additional_features': additional_features
}

with open('retailrocket_dataset.pkl', 'wb') as f:
    pickle.dump(save_obj, f)

print("✅ 데이터셋 저장 완료: retailrocket_dataset.pkl")
print(f"✔ Length of session_event_sequences: {len(session_event_sequences)}")
print(f"✔ Length of time_to_next_session: {len(time_to_next_session)}")
print(f"✔ Shape of additional_features: {additional_features.shape}")
