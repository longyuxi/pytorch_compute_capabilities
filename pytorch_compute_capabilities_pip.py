"""
Analyze all PyTorch 2.x versions and generate comprehensive table. Note the inclusion of 'manylinux_2_28_x86_64' is release name will only find 2.7.0 and newer (currently through 2.8.0).
"""

from pathlib import Path
import subprocess
import tempfile
from typing import Any
import zipfile
import re

import requests


def get_pypi_package_info(
    package_name: str, version: str | None = None
) -> dict[str, Any]:
    """
    Fetch package information from PyPI API.

    Args:
        package_name: Name of the package (e.g., 'torch')
        version: Specific version to fetch, if None fetches latest

    Returns:
        Dictionary containing package information
    """
    if version:
        url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    else:
        url = f"https://pypi.org/pypi/{package_name}/json"

    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def extract_python_version_from_filename(filename: str) -> str:
    """
    Extract Python version from wheel filename.

    Args:
        filename: Wheel filename (e.g., 'torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl')

    Returns:
        Python version string (e.g., '3.13')
    """
    # Wheel filename format: {package}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl
    parts = filename.split("-")
    if len(parts) >= 3:
        python_tag = parts[2]  # e.g., 'cp313', 'cp39'
        if python_tag.startswith("cp"):
            version_num = python_tag[2:]  # Remove 'cp' prefix
            if len(version_num) >= 2:
                major = version_num[0]
                minor = version_num[1:]
                return f"{major}.{minor}"
    return "unknown"


def get_wheel_download_links(package_name: str, version: str) -> list[dict[str, str]]:
    """
    Get wheel file download links for a specific PyTorch version.
    Only returns manylinux_2_28_x86_64 wheels.

    Args:
        package_name: Name of the package (e.g., 'torch')
        version: Version string (e.g., '2.8.0')

    Returns:
        List of dictionaries containing wheel file information
    """
    try:
        package_info = get_pypi_package_info(package_name, version)

        wheels = []
        for file_info in package_info["urls"]:
            if file_info["packagetype"] == "bdist_wheel":
                filename = file_info["filename"]

                # Only include manylinux_2_28_x86_64 wheels
                if "manylinux_2_28_x86_64" not in filename:
                    continue

                python_version = extract_python_version_from_filename(filename)

                wheel_info = {
                    "filename": filename,
                    "url": file_info["url"],
                    "size": file_info["size"],
                    "python_version": python_version,
                    "platform_tag": "manylinux_2_28_x86_64",
                }
                wheels.append(wheel_info)

        # Sort by Python version for consistent ordering
        wheels.sort(key=lambda x: x["python_version"])
        return wheels

    except requests.exceptions.RequestException as e:
        print(f"Error fetching package info: {e}")
        return []
    except KeyError as e:
        print(f"Error parsing package info: {e}")
        return []


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def print_wheel_info(wheels: list[dict[str, str]]) -> None:
    """Print wheel information in a readable format."""
    for wheel in wheels:
        size_str = format_file_size(wheel["size"])
        print(f"{wheel['filename']} ({size_str})")
        print(f"  URL: {wheel['url']}")
        print(f"  Python: {wheel['python_version']}")
        print(f"  Platform: {wheel['platform_tag']}")
        print()


def download_wheel(url: str, filename: str, download_dir: Path) -> Path:
    """
    Download a wheel file from URL.

    Args:
        url: URL to download from
        filename: Name of the file
        download_dir: Directory to save the file

    Returns:
        Path to the downloaded file
    """
    download_path = download_dir / filename

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    print(f"\nDownloaded to {download_path}")
    return download_path


def extract_wheel(wheel_path: Path, extract_dir: Path) -> Path:
    """
    Extract wheel file as a zip archive.

    Args:
        wheel_path: Path to the wheel file
        extract_dir: Directory to extract to

    Returns:
        Path to the extraction directory
    """
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


def get_cuda_architectures(extract_dir: Path) -> list[str]:
    """
    Run cuobjdump on libtorch_cuda.so and extract supported architectures.

    Args:
        extract_dir: Directory where wheel was extracted

    Returns:
        List of supported CUDA architectures
    """
    libtorch_path = extract_dir / "torch" / "lib" / "libtorch_cuda.so"

    if not libtorch_path.exists():
        print(f"Warning: {libtorch_path} not found")
        return []

    try:
        cuobjdump_cmd = "cuobjdump"
        # Example: cuobjdump_cmd = "singularity exec --bind /path/to/bind_dir /path/to/cuda.sif cuobjdump"
        command_raw = f"{cuobjdump_cmd} '{libtorch_path}'"

        print(f"Running: {command_raw}")
        result_raw = subprocess.run(
            command_raw,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            executable="/bin/bash",
        )

        print(f"cuobjdump output length: {len(result_raw.stdout)} characters")

        # Let's look for lines containing 'arch' (case insensitive)
        arch_lines = []
        for line in result_raw.stdout.split("\n"):
            if "arch" in line.lower():
                arch_lines.append(line.strip())

        if arch_lines:
            # Sort and remove duplicates
            unique_archs = sorted(set(arch_lines))
            print("Found architectures:")
            for arch in unique_archs:
                print(f"  {arch}")
            return unique_archs
        else:
            # Let's see if there are any lines that might contain architecture info
            print("No lines with 'arch' found. Looking for other patterns...")

            # Look for sm_ patterns
            sm_lines = []
            for line in result_raw.stdout.split("\n"):
                if "sm_" in line.lower():
                    sm_lines.append(line.strip())

            if sm_lines:
                print("Found lines with 'sm_' pattern:")
                for line in sm_lines[:10]:  # Show first 10 matches
                    print(f"  {line}")
                return sm_lines
            else:
                print(
                    "No architecture patterns found. Showing first 20 lines of cuobjdump output:"
                )
                lines = result_raw.stdout.split("\n")
                for i, line in enumerate(lines[:20]):
                    print(f"  {i + 1}: {line}")
                return []

    except subprocess.CalledProcessError as e:
        print(f"Error running cuobjdump command: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def analyze_first_wheel(wheels: list[dict[str, str]]) -> dict[str, Any] | None:
    """
    Download and analyze the first wheel file to extract CUDA architectures.

    Args:
        wheels: List of wheel information

    Returns:
        Dictionary with wheel info and supported architectures
    """
    if not wheels:
        print("No wheels to analyze")
        return None

    first_wheel = wheels[0]
    print(f"Analyzing first wheel: {first_wheel['filename']}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download the wheel
        wheel_path = download_wheel(
            first_wheel["url"], first_wheel["filename"], temp_path
        )

        # Extract the wheel
        extract_dir = temp_path / "extracted"
        extract_dir.mkdir()
        extract_wheel(wheel_path, extract_dir)

        # Get CUDA architectures
        archs = get_cuda_architectures(extract_dir)

        return {"wheel_info": first_wheel, "cuda_architectures": archs}


def analyze_all_wheels(
    wheels: list[dict[str, str]], package_version: str
) -> list[dict[str, Any]]:
    """
    Download and analyze all wheel files to extract CUDA architectures.

    Args:
        wheels: List of wheel information
        package_version: Version string (e.g., '2.8.0')

    Returns:
        List of dictionaries with wheel info and supported architectures
    """
    results = []

    for i, wheel in enumerate(wheels, 1):
        print(f"\n{'=' * 60}")
        print(f"Analyzing wheel {i}/{len(wheels)}: {wheel['filename']}")
        print(f"{'=' * 60}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Download the wheel
                wheel_path = download_wheel(wheel["url"], wheel["filename"], temp_path)

                # Extract the wheel
                extract_dir = temp_path / "extracted"
                extract_dir.mkdir()
                extract_wheel(wheel_path, extract_dir)

                # Get CUDA architectures
                archs = get_cuda_architectures(extract_dir)

                # Clean up arch strings to just extract sm_XX values
                clean_archs = []
                for arch in archs:
                    if "sm_" in arch:
                        # Extract just the sm_XX part

                        match = re.search(r"sm_\d+[a-z]*", arch)
                        if match:
                            clean_archs.append(match.group())

                results.append(
                    {
                        "wheel_info": wheel,
                        "cuda_architectures": sorted(set(clean_archs)),
                        "package_version": package_version,
                    }
                )

                print(f"✓ Successfully analyzed {wheel['filename']}")
                if clean_archs:
                    print(f"  Architectures: {', '.join(sorted(set(clean_archs)))}")
                else:
                    print("  No CUDA architectures found")

            except Exception as e:
                print(f"✗ Error analyzing {wheel['filename']}: {e}")
                results.append(
                    {
                        "wheel_info": wheel,
                        "cuda_architectures": [],
                        "package_version": package_version,
                    }
                )

    return results


def generate_pip_table(
    package: str, version: str, results: list[dict[str, Any]]
) -> str:
    """
    Generate a markdown table in the same format as table.md.

    Args:
        package: Package name (e.g., 'torch')
        version: Version string (e.g., '2.8.0')
        results: List of analysis results

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| package | architectures |")
    lines.append("|---------|---------------|")

    for result in results:
        wheel_info = result["wheel_info"]
        archs = result["cuda_architectures"]

        # Use the actual wheel filename
        package_name = wheel_info["filename"]

        # Format architectures
        if archs:
            arch_str = ", ".join(archs)
        else:
            arch_str = ""

        lines.append(f"| {package_name} | {arch_str} |")

    return "\n".join(lines)


def get_all_pytorch_2x_versions(package_name: str = "torch") -> list[str]:
    """
    Get all PyTorch 2.x versions available on PyPI.

    Args:
        package_name: Package name (default: 'torch')

    Returns:
        List of version strings (e.g., ['2.0.0', '2.0.1', '2.1.0', ...])
    """
    try:
        package_info = get_pypi_package_info(package_name)
        all_versions = list(package_info["releases"].keys())

        # Filter for 2.x versions and sort them
        pytorch_2x_versions = []
        for version in all_versions:
            if version.startswith("2."):
                # Skip pre-release versions (rc, dev, etc.)
                if not any(marker in version for marker in ["rc", "dev", "a", "b"]):
                    pytorch_2x_versions.append(version)

        # Sort versions using semantic versioning
        def version_key(v):
            parts = v.split(".")
            return [int(x) for x in parts]

        pytorch_2x_versions.sort(key=version_key, reverse=True)  # Latest first
        return pytorch_2x_versions

    except Exception as e:
        print(f"Error getting PyTorch versions: {e}")
        return []


def save_table_to_file(table_content: str, filename: str = "table_pip.md") -> None:
    """
    Save the markdown table to a file.

    Args:
        table_content: The markdown table content
        filename: Output filename
    """
    with open(filename, "w") as f:
        f.write(table_content)
    print(f"Table saved to {filename}")


def generate_comprehensive_pip_table(all_results: list[dict[str, Any]]) -> str:
    """
    Generate a comprehensive markdown table for all versions and wheels.
    Sorted with newest versions first, then by Python version.

    Args:
        all_results: List of all analysis results across versions

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| package | architectures |")
    lines.append("|---------|---------------|")

    # Sort results: newest version first, then by python version
    def sort_key(result):
        version = result["package_version"]
        python_version = result["wheel_info"]["python_version"]

        # Parse version for proper sorting (e.g., "2.8.0" -> [2, 8, 0])
        version_parts = [int(x) for x in version.split(".")]

        # Parse python version (e.g., "3.10" -> [3, 10])
        python_parts = [int(x) for x in python_version.split(".")]

        # Return tuple: (negative version for desc order, python version for asc order)
        return ([-x for x in version_parts], python_parts)

    sorted_results = sorted(all_results, key=sort_key)

    for result in sorted_results:
        wheel_info = result["wheel_info"]
        archs = result["cuda_architectures"]

        # Use the actual wheel filename
        package_name = wheel_info["filename"]

        # Format architectures
        if archs:
            arch_str = ", ".join(archs)
        else:
            arch_str = ""

        lines.append(f"| {package_name} | {arch_str} |")

    return "\n".join(lines)


def main():
    """Main function to analyze all PyTorch 2.x versions and generate comprehensive table."""
    package = "torch"

    # Get all PyTorch 2.x versions
    print("Fetching all PyTorch 2.x versions from PyPI...")
    versions = get_all_pytorch_2x_versions(package)

    if not versions:
        print("No PyTorch 2.x versions found!")
        return

    print(f"Found {len(versions)} PyTorch 2.x versions:")
    for i, version in enumerate(versions):
        print(f"  {i + 1:2}. {version}")
    print()

    # Count total wheels before processing
    print("Counting total wheels to be processed...")
    total_wheels_estimate = 0
    version_wheel_counts = {}

    for version in versions:
        wheels = get_wheel_download_links(package, version)
        wheel_count = len(wheels)
        version_wheel_counts[version] = wheel_count
        total_wheels_estimate += wheel_count
        print(f"  {version}: {wheel_count} wheels")

    print(f"\nTotal wheels to process: {total_wheels_estimate}")

    if total_wheels_estimate == 0:
        print("No wheels found to process!")
        return

    # Estimate download size (assuming ~850MB per wheel based on torch 2.8.0)
    estimated_size_gb = (total_wheels_estimate * 850) / 1024
    print(f"Estimated download size: ~{estimated_size_gb:.1f} GB")
    print("This will take significant time and bandwidth!")
    print()

    # Ask for confirmation
    try:
        confirm = input("Do you want to proceed? (y/N): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("Aborted.")
            return
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    all_results = []
    total_wheels = 0

    # Process each version
    for version_idx, version in enumerate(versions, 1):
        wheel_count = version_wheel_counts[version]

        if wheel_count == 0:
            print(f"Skipping {version} (no manylinux_2_28_x86_64 wheels)")
            continue

        print(f"\n{'=' * 80}")
        print(
            f"Processing version {version_idx}/{len(versions)}: {package} {version} ({wheel_count} wheels)"
        )
        print(f"{'=' * 80}")

        wheels = get_wheel_download_links(package, version)
        total_wheels += len(wheels)

        # Analyze all wheels for this version
        version_results = analyze_all_wheels(wheels, version)
        all_results.extend(version_results)

        # Print summary for this version
        print(f"\nSummary for {version}:")
        for result in version_results:
            wheel_info = result["wheel_info"]
            archs = result["cuda_architectures"]
            python_ver = wheel_info["python_version"]
            arch_count = len(archs)
            print(
                f"  Python {python_ver}: {arch_count} architectures - {', '.join(archs) if archs else 'None'}"
            )

    # Generate comprehensive table
    if all_results:
        print(f"\n{'=' * 80}")
        print(f"Generating comprehensive table for all {len(versions)} versions...")
        print(f"Total wheels processed: {total_wheels}")
        print(f"{'=' * 80}")

        table_content = generate_comprehensive_pip_table(all_results)
        save_table_to_file(table_content)

        # Final summary
        version_counts = {}
        for result in all_results:
            version = result["package_version"]
            if version not in version_counts:
                version_counts[version] = 0
            version_counts[version] += 1

        print("\nFinal Summary:")
        print(f"Total PyTorch versions processed: {len(version_counts)}")
        print(f"Total wheel files analyzed: {len(all_results)}")
        for version, count in sorted(version_counts.items(), reverse=True):
            print(f"  {version}: {count} wheels")

    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
