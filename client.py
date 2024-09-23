# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict", 
    json={
    "input": {
        "battery_power": 1224,
        "blue": 1,
        "clock_speed": 2.4,
        "dual_sim": 0,
        "fc": 16,
        "four_g": 1,
        "int_memory": 64,
        "m_dep": 0.3,
        "mobile_wt": 113,
        "n_cores": 8,
        "pc": 15,
        "px_height": 1350,
        "px_width": 785,
        "ram": 12,
        "sc_h": 7,
        "sc_w": 5,
        "talk_time": 1,
        "three_g": 1,
        "touch_screen": 1,
        "wifi": 1
    }
}
)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
