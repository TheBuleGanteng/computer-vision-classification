"""
Test utility for RunPod FastAPI endpoints

This module provides functions to test all RunPod FastAPI endpoints to ensure
they are working properly before attempting actual file downloads.
"""

import requests
import time
from typing import Dict, Any, Optional
from utils.logger import logger


def test_health_endpoint(worker_endpoint: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Test the /health endpoint to verify FastAPI server is running.

    Args:
        worker_endpoint: RunPod worker endpoint URL
        timeout: Request timeout in seconds

    Returns:
        Dictionary with test results
    """
    test_result = {
        "endpoint": "/health",
        "url": f"{worker_endpoint}/health",
        "success": False,
        "status_code": None,
        "response": None,
        "error": None,
        "response_time": None
    }

    start_time = time.time()
    try:
        logger.info(f"TEST HEALTH ENDPOINT: {test_result['url']}")

        response = requests.get(test_result["url"], timeout=timeout)
        test_result["response_time"] = time.time() - start_time
        test_result["status_code"] = response.status_code

        if response.status_code == 200:
            test_result["response"] = response.json()
            test_result["success"] = True
            logger.info(f"HEALTH ENDPOINT SUCCESS: {test_result['response']}")
        else:
            test_result["response"] = response.text
            test_result["error"] = f"HTTP {response.status_code}"
            logger.error(f"HEALTH ENDPOINT FAILED: {response.status_code} - {response.text}")

    except Exception as e:
        test_result["error"] = str(e)
        test_result["response_time"] = time.time() - start_time
        logger.error(f"HEALTH ENDPOINT ERROR: {e}")

    return test_result


def test_files_endpoint(worker_endpoint: str, run_name: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Test the /files/{run_name} endpoint to list available files.

    Args:
        worker_endpoint: RunPod worker endpoint URL
        run_name: Run name to query for files
        timeout: Request timeout in seconds

    Returns:
        Dictionary with test results
    """
    test_result = {
        "endpoint": "/files/{run_name}",
        "url": f"{worker_endpoint}/files/{run_name}",
        "success": False,
        "status_code": None,
        "response": None,
        "error": None,
        "response_time": None,
        "files_found": 0
    }

    start_time = time.time()
    try:
        logger.info(f"TEST FILES ENDPOINT: {test_result['url']}")

        response = requests.get(test_result["url"], timeout=timeout)
        test_result["response_time"] = time.time() - start_time
        test_result["status_code"] = response.status_code

        if response.status_code == 200:
            response_data = response.json()
            test_result["response"] = response_data
            test_result["files_found"] = len(response_data.get("files", []))
            test_result["success"] = True
            logger.info(f"FILES ENDPOINT SUCCESS: Found {test_result['files_found']} files")

            # Log first few files for debugging
            files = response_data.get("files", [])
            if files:
                logger.info(f"First 5 files: {files[:5]}")
            else:
                logger.warning(f"No files found for run: {run_name}")

        elif response.status_code == 404:
            test_result["response"] = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            test_result["error"] = f"Run '{run_name}' not found (404)"
            logger.warning(f"FILES ENDPOINT 404: Run '{run_name}' not found")
        else:
            test_result["response"] = response.text
            test_result["error"] = f"HTTP {response.status_code}"
            logger.error(f"FILES ENDPOINT FAILED: {response.status_code} - {response.text}")

    except Exception as e:
        test_result["error"] = str(e)
        test_result["response_time"] = time.time() - start_time
        logger.error(f"FILES ENDPOINT ERROR: {e}")

    return test_result


def test_download_endpoint(worker_endpoint: str, run_name: str, file_path: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Test the /download/{run_name}/{file_path} endpoint to download a specific file.

    Args:
        worker_endpoint: RunPod worker endpoint URL
        run_name: Run name containing the file
        file_path: Relative path to the file
        timeout: Request timeout in seconds

    Returns:
        Dictionary with test results
    """
    test_result = {
        "endpoint": "/download/{run_name}/{file_path}",
        "url": f"{worker_endpoint}/download/{run_name}/{file_path}",
        "success": False,
        "status_code": None,
        "response": None,
        "error": None,
        "response_time": None,
        "content_length": 0,
        "content_type": None
    }

    start_time = time.time()
    try:
        logger.info(f"TEST DOWNLOAD ENDPOINT: {test_result['url']}")

        response = requests.get(test_result["url"], timeout=timeout, stream=True)
        test_result["response_time"] = time.time() - start_time
        test_result["status_code"] = response.status_code
        test_result["content_type"] = response.headers.get('content-type')

        if response.status_code == 200:
            # Don't download the full content, just check headers
            test_result["content_length"] = int(response.headers.get('content-length', 0))
            test_result["success"] = True
            logger.info(f"DOWNLOAD ENDPOINT SUCCESS: {test_result['content_length']} bytes, type: {test_result['content_type']}")
        elif response.status_code == 404:
            test_result["response"] = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
            test_result["error"] = f"File '{file_path}' not found in run '{run_name}' (404)"
            logger.warning(f"DOWNLOAD ENDPOINT 404: File '{file_path}' not found")
        else:
            test_result["response"] = response.text
            test_result["error"] = f"HTTP {response.status_code}"
            logger.error(f"DOWNLOAD ENDPOINT FAILED: {response.status_code} - {response.text}")

    except Exception as e:
        test_result["error"] = str(e)
        test_result["response_time"] = time.time() - start_time
        logger.error(f"DOWNLOAD ENDPOINT ERROR: {e}")

    return test_result


def test_plots_directory_structure(worker_endpoint: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Test the /debug/plots-structure endpoint to see directory structure.

    Args:
        worker_endpoint: RunPod worker endpoint URL
        timeout: Request timeout in seconds

    Returns:
        Dictionary with test results
    """
    test_result = {
        "endpoint": "/debug/plots-structure",
        "url": f"{worker_endpoint}/debug/plots-structure",
        "success": False,
        "status_code": None,
        "response": None,
        "error": None,
        "response_time": None
    }

    start_time = time.time()
    try:
        logger.info(f"TEST PLOTS DIRECTORY STRUCTURE: {test_result['url']}")

        response = requests.get(test_result["url"], timeout=timeout)
        test_result["response_time"] = time.time() - start_time
        test_result["status_code"] = response.status_code

        if response.status_code == 200:
            test_result["response"] = response.json()
            test_result["success"] = True
            logger.info(f"PLOTS STRUCTURE SUCCESS: {test_result['response']}")
        else:
            test_result["response"] = response.text
            test_result["error"] = f"Debug endpoint not available (HTTP {response.status_code})"
            logger.warning(f"PLOTS STRUCTURE ENDPOINT NOT AVAILABLE: {response.status_code}")

    except Exception as e:
        test_result["error"] = str(e)
        test_result["response_time"] = time.time() - start_time
        logger.warning(f"PLOTS STRUCTURE ERROR (expected if no debug endpoint): {e}")

    return test_result


def run_comprehensive_endpoint_tests(worker_endpoint: str, run_name: str, sample_file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run all endpoint tests and return comprehensive results.

    Args:
        worker_endpoint: RunPod worker endpoint URL
        run_name: Run name to test with
        sample_file_path: Optional file path to test download with

    Returns:
        Dictionary with all test results
    """
    logger.info(f"========== COMPREHENSIVE RUNPOD ENDPOINT TESTING ==========")
    logger.info(f"Worker Endpoint: {worker_endpoint}")
    logger.info(f"Run Name: {run_name}")

    results = {
        "worker_endpoint": worker_endpoint,
        "run_name": run_name,
        "test_timestamp": time.time(),
        "tests": {}
    }

    # Test 1: Health endpoint
    results["tests"]["health"] = test_health_endpoint(worker_endpoint)

    # Test 2: Files listing endpoint
    results["tests"]["files"] = test_files_endpoint(worker_endpoint, run_name)

    # Test 3: Download endpoint (if we have files or a sample file path)
    if sample_file_path:
        results["tests"]["download"] = test_download_endpoint(worker_endpoint, run_name, sample_file_path)
    elif results["tests"]["files"]["success"] and results["tests"]["files"]["files_found"] > 0:
        # Use the first available file for download test
        first_file = results["tests"]["files"]["response"]["files"][0]
        results["tests"]["download"] = test_download_endpoint(worker_endpoint, run_name, first_file)
    else:
        logger.warning(f"Skipping download test - no files available or sample file provided")
        results["tests"]["download"] = {"skipped": True, "reason": "No files available"}

    # Test 4: Directory structure (optional)
    results["tests"]["plots_structure"] = test_plots_directory_structure(worker_endpoint)

    # Summary
    successful_tests = sum(1 for test in results["tests"].values()
                          if isinstance(test, dict) and test.get("success", False))
    total_tests = len([test for test in results["tests"].values()
                      if isinstance(test, dict) and not test.get("skipped", False)])

    results["summary"] = {
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "success_rate": (successful_tests / total_tests) if total_tests > 0 else 0.0,
        "all_critical_tests_passed": (
            results["tests"]["health"]["success"] and
            results["tests"]["files"]["success"]
        )
    }

    logger.info(f"========== ENDPOINT TESTING COMPLETE ==========")
    logger.info(f"Success Rate: {successful_tests}/{total_tests} ({results['summary']['success_rate']:.1%})")
    logger.info(f"Critical Tests (Health + Files): {'PASSED' if results['summary']['all_critical_tests_passed'] else 'FAILED'}")

    return results


def log_endpoint_test_summary(test_results: Dict[str, Any]) -> None:
    """
    Log a detailed summary of endpoint test results.

    Args:
        test_results: Results from run_comprehensive_endpoint_tests
    """
    logger.info(f"========== ENDPOINT TEST SUMMARY ==========")

    for test_name, test_data in test_results["tests"].items():
        if test_data.get("skipped"):
            logger.info(f"{test_name.upper()}: SKIPPED - {test_data.get('reason', 'Unknown')}")
            continue

        status = "PASS" if test_data.get("success") else "FAIL"
        response_time = test_data.get("response_time")
        time_str = f" ({response_time:.2f}s)" if response_time else ""

        logger.info(f"{status} {test_name.upper()}{time_str}")

        if test_data.get("error"):
            logger.info(f"   Error: {test_data['error']}")
        elif test_name == "files" and test_data.get("success"):
            logger.info(f"   Files found: {test_data.get('files_found', 0)}")
        elif test_name == "download" and test_data.get("success"):
            logger.info(f"   Content: {test_data.get('content_length', 0)} bytes")

    logger.info(f"========================================")