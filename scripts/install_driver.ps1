Set-Location C:\workspace
Stop-Process -Name "nvidia_driver" -Force -ErrorAction SilentlyContinue
Start-Sleep 3
if (Test-Path "C:\workspace\nvidia_driver.exe") {
    Write-Host "Installing NVIDIA driver silently..."
    Start-Process -FilePath "C:\workspace\nvidia_driver.exe" -ArgumentList "-s","-noreboot" -Wait
    Write-Host "DRIVER_INSTALL_DONE"
} else {
    Write-Host "DRIVER_FILE_NOT_FOUND"
}
