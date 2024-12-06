import aiohttp
import asyncio
import os

# BASE_URL = "https://ransahai-crud-production.pyshivam.workers.dev/api/v1"
BASE_URL = "http://localhost:8787/api/v1"

class ApiClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.secret = os.getenv("SECRET")

    async def fetch(self, endpoint: str, method: str = "GET", data: dict = None):
        headers = {
            "Authorization": f"Bearer {self.secret}"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            url = f"{self.base_url}{endpoint}"
            async with session.request(method, url, json=data) as response:
                response.raise_for_status()
                return await response.json()

    async def get_job_ids(self):
        return await self.fetch("/public/jobs")

    async def get_job_details(self, job_id: str):
        return await self.fetch(f"/public/jobs/{job_id}")

    async def get_candidate_details(self, candidate_id: str):
        return await self.fetch(f"/internals/get-interview-details?interviewId={candidate_id}")

# Example usage
async def main():
    client = ApiClient(BASE_URL)
    job_ids = await client.get_job_ids()
    print("Job IDs:", job_ids)

    if job_ids and job_ids.get("data"):
        job_id = job_ids["data"][0]["id"]
        candidate_details = await client.get_job_details(job_id)
        print("Candidate Details:", candidate_details)

if __name__ == "__main__":
    asyncio.run(main())